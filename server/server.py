import argparse
import csv
import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import grpc
import numpy as np
from grpc import aio

import drone_pb2, drone_pb2_grpc

# Two-cone detector
from cone_detector_2stage import (
    TwoConePreselector,
    TwoConeGPSEstimator,
    _parse_manual_start,
    _calc_ts_iso,
)

# Four-cone detector
from four_cone_detector import SimpleRefPairEstimator


# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------
def create_tracker():
    for ctor in (
        getattr(cv2, "legacy", None)
        and getattr(cv2.legacy, "TrackerKCF_create", None),
        getattr(cv2, "TrackerKCF_create", None),
        getattr(cv2, "TrackerCSRT_create", None),
        getattr(cv2, "TrackerMOSSE_create", None),
    ):
        if ctor:
            try:
                return ctor()
            except Exception:
                continue
    raise RuntimeError("No available OpenCV tracker found.")


def encode_jpg(img, quality=70):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


@dataclass
class SharedState:
    need_roi: bool = False
    box: Optional[Tuple[int, int, int, int]] = None
    stop: bool = False
    has_tracker: bool = False


# ----------------------------------------------------------
# Main Servicer
# ----------------------------------------------------------
class DroneStreamServicer(drone_pb2_grpc.DroneStreamServicer):
    def __init__(
        self,
        video_path,
        cones_gps_path,
        output_csv,
        start_time,
        cone_spacing_m=3.5,
        sample_stride=10,
        min_frames=40,
        emit_stride=2,
        cones_layout="24",
        refA=None,
        refB=None,
        cone_spacing_vertical_m=None,
        cone_spacing_horizontal_m=None,
    ):
        self.video_path = video_path
        self.cones_gps_path = cones_gps_path
        self.output_csv = output_csv
        self.start_time = start_time
        self.cone_spacing_m = cone_spacing_m
        self.sample_stride = sample_stride
        self.min_frames = min_frames
        self.emit_stride = max(1, emit_stride)

        self.cones_layout = cones_layout
        self.refA = refA
        self.refB = refB

        self.cone_spacing_vertical_m = cone_spacing_vertical_m
        self.cone_spacing_horizontal_m = cone_spacing_horizontal_m

    # ------------------------------------------------------
    # gRPC method
    # ------------------------------------------------------
    async def StreamVideo(self, request_iterator, context):
        state = SharedState()
        roi_frame_sent = False

        # ------------------------------------------
        # Stage 1: Select reference cones
        # ------------------------------------------
        if self.cones_layout == "4":
            keyA, keyB = self.refA, self.refB
            yield drone_pb2.Frame(status=f"Stage1: four-cone mode A={keyA}, B={keyB}")
        else:
            selector = TwoConePreselector(
                sample_stride=self.sample_stride, min_frames=self.min_frames
            )
            picked = selector.scan_video(self.video_path, start_time=self.start_time)
            if picked is None:
                yield drone_pb2.Frame(status="ERROR: Stage1 failed (cannot select cones)")
                return
            keyA, keyB, summary = picked
            yield drone_pb2.Frame(
                status=f"Stage1: two-cone auto-selected A={keyA}, B={keyB} {summary}"
            )

        # ------------------------------------------
        # Open video
        # ------------------------------------------
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            yield drone_pb2.Frame(status=f"ERROR: Cannot open video {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        t0 = _parse_manual_start(self.start_time)

        ok, first_frame = cap.read()
        if not ok:
            yield drone_pb2.Frame(status="ERROR: Cannot read first frame")
            return

        paused_frame = first_frame.copy()

        # ------------------------------------------
        # Force ROI mode initially (BOTH 24 & 4 cones)
        # ------------------------------------------
        state.need_roi = True
        roi_frame_sent = False
        state.box = None
        state.has_tracker = False

        # ------------------------------------------
        # Initialize estimator
        # ------------------------------------------
        if self.cones_layout == "4":
            spacing_v = self.cone_spacing_vertical_m or self.cone_spacing_m
            spacing_h = self.cone_spacing_horizontal_m or self.cone_spacing_m

            estimator = SimpleRefPairEstimator(
                self.cones_gps_path,
                keyA, keyB,
                cone_spacing_vertical_m=spacing_v,
                cone_spacing_horizontal_m=spacing_h,
            )
            print(f"[INFO] Four-cone estimator: vertical={spacing_v}, horizontal={spacing_h}")

        else:
            estimator = TwoConeGPSEstimator(
                self.cones_gps_path, keyA, keyB, cone_spacing_m=self.cone_spacing_m
            )
            print(f"[INFO] Two-cone estimator: spacing={self.cone_spacing_m}")

        results = []

        # ------------------------------------------
        # Notify ready
        # ------------------------------------------
        jpeg0 = encode_jpg(first_frame, 80)
        if jpeg0:
            yield drone_pb2.Frame(jpg=jpeg0, status="Ready", hint="Press r to select ROI")

        frame_idx = 1
        emit_counter = 0
        tracker = None

        # ==========================================
        # Control consumer
        # ==========================================
        async def control_consumer():
            nonlocal roi_frame_sent
            async for ctrl in request_iterator:

                # Key events
                if ctrl.HasField("key"):
                    key = ctrl.key.key.strip()

                    if key == "r":
                        print("[CTRL] ROI requested by client")
                        state.need_roi = True
                        state.has_tracker = False
                        state.box = None
                        roi_frame_sent = False

                    elif key == "q":
                        print("[CTRL] Quit requested by client")
                        state.stop = True
                        break

                # ROI from client
                elif ctrl.HasField("roi"):
                    x, y, w, h = int(ctrl.roi.x), int(ctrl.roi.y), int(ctrl.roi.w), int(ctrl.roi.h)
                    print(f"[CTRL] ROI received: {x},{y},{w},{h}")

                    state.box = (x, y, w, h)
                    state.need_roi = False
                    state.has_tracker = False
                    roi_frame_sent = False

            state.stop = True

        ctrl_task = asyncio.create_task(control_consumer())

        # ==========================================
        # MAIN LOOP
        # ==========================================
        try:
            while not state.stop:

                # ----------------------------------------------------------
                # ROI MODE — Freeze at paused_frame
                # ----------------------------------------------------------
                if state.need_roi:
                    if not roi_frame_sent:
                        jpg = encode_jpg(paused_frame, 75)
                        if jpg:
                            print("[SERVER] Sending ROI freeze frame")
                            yield drone_pb2.Frame(
                                jpg=jpg,
                                status="ROI requested",
                                hint="ROI mode (drag + Enter)"
                            )
                        roi_frame_sent = True

                    await asyncio.sleep(0.05)
                    continue

                # ----------------------------------------------------------
                # NORMAL MODE — Read next frame
                # ----------------------------------------------------------
                ok, frame = cap.read()
                if not ok:
                    print("[SERVER] Video finished")
                    break

                frame_idx += 1
                paused_frame = frame.copy()
                vis = frame

                # ----------------------------------------------------------
                # TRACKER INIT after ROI
                # ----------------------------------------------------------
                if not state.has_tracker and state.box is not None:
                    print("[SERVER] Initializing tracker:", state.box)
                    tracker = create_tracker()
                    tracker.init(frame, tuple(state.box))
                    state.has_tracker = True

                # ----------------------------------------------------------
                # TRACKER UPDATE (Robust)
                # ----------------------------------------------------------
                if state.has_tracker and tracker is not None:
                    try:
                        result = tracker.update(frame)
                    except Exception as e:
                        print(f"[WARN] tracker.update crashed: {e}")
                        result = None

                    success, box_new = False, None

                    if isinstance(result, tuple):

                        # Format: (success, (x,y,w,h))
                        if len(result) == 2 and isinstance(result[1], (tuple, list)):
                            success, box_new = result

                        # Format: (success, x, y, w, h)
                        elif len(result) >= 5:
                            success = bool(result[0])
                            box_new = result[1:5]

                        # Format: (success,)
                        elif len(result) == 1:
                            success = bool(result[0])

                    elif isinstance(result, bool):
                        success = result

                    # ------------------------------------------------------
                    # SUCCESS
                    # ------------------------------------------------------
                    if success and box_new is not None:
                        x, y, w, h = map(int, box_new[:4])
                        cx, cy = x + w // 2, y + h // 2
                        state.box = (x, y, w, h)

                        # ------------------------------------------
                        # FIX: Extract lat/lon ONLY from estimator
                        # ------------------------------------------
                        out = estimator.estimate_frame(frame, (cx, cy))

                        if out is not None:
                            # Case 1: (lat, lon)
                            if isinstance(out, (tuple, list)) and len(out) == 2:
                                lat, lon = out

                            # Case 2: (lat, lon, flag)
                            elif isinstance(out, (tuple, list)) and len(out) == 3:
                                lat, lon, _ = out

                            # Case 3: ((lat,lon), flag)
                            elif (
                                isinstance(out, (tuple, list))
                                and len(out) == 2
                                and isinstance(out[0], (tuple, list))
                            ):
                                lat, lon = out[0]

                            else:
                                print("[WARN] Unknown estimator output:", out)
                                lat = lon = None

                            if lat is not None:
                                ts_iso = _calc_ts_iso(cap, frame_idx, fps, t0)
                                results.append((ts_iso, lat, lon))

                        # Draw box
                        vis = frame.copy()
                        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
                        cv2.circle(vis, (cx, cy), 3, (0,0,255), -1)
                        cv2.putText(
                            vis,
                            f"frame:{frame_idx} tracks:{len(results)} (r:ROI q:exit)",
                            (20,30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(255,255,255),2
                        )

                    # ------------------------------------------------------
                    # FAILURE → Request ROI
                    # ------------------------------------------------------
                    else:
                        print("[SERVER] Tracking lost → entering ROI mode")
                        yield drone_pb2.Frame(status="tracking_lost")

                        state.need_roi = True
                        state.has_tracker = False
                        state.box = None
                        tracker = None
                        roi_frame_sent = False

                        paused_frame = frame.copy()
                        await asyncio.sleep(0.05)
                        continue

                # ----------------------------------------------------------
                # SEND FRAME TO CLIENT
                # ----------------------------------------------------------
                emit_counter += 1
                if emit_counter % self.emit_stride == 0:
                    jpg = encode_jpg(vis, 70)
                    if jpg:
                        yield drone_pb2.Frame(jpg=jpg)

                await asyncio.sleep(0.002)

        # ======================================================
        # CLEANUP
        # ======================================================
        finally:
            if self.output_csv and len(results) > 0:
                with open(self.output_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp", "latitude", "longitude"])
                    w.writerows(results)
                yield drone_pb2.Frame(status=f"Saved CSV: {self.output_csv} ({len(results)} rows)")

            cap.release()
            yield drone_pb2.Frame(status="Stopped.")

            ctrl_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ctrl_task


# ----------------------------------------------------------
# Server Runner
# ----------------------------------------------------------
async def serve_async(args):
    server = aio.server()

    drone_pb2_grpc.add_DroneStreamServicer_to_server(
        DroneStreamServicer(
            video_path=args.video,
            cones_gps_path=args.cones_gps,
            output_csv=args.output_csv,
            start_time=args.start_time,
            cone_spacing_m=args.cone_spacing_m,
            sample_stride=args.sample_stride,
            min_frames=args.min_frames,
            emit_stride=args.emit_stride,
            cones_layout=args.cones_layout,
            refA=args.refA,
            refB=args.refB,
            cone_spacing_vertical_m=args.cone_spacing_vertical_m,
            cone_spacing_horizontal_m=args.cone_spacing_horizontal_m,
        ),
        server,
    )

    server.add_insecure_port(f"{args.host}:{args.port}")

    print(f"[SERVER] gRPC server listening on {args.host}:{args.port}")
    await server.start()
    await server.wait_for_termination()


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--cones-gps", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--start-time", default=None)

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)

    parser.add_argument("--cone-spacing-m", type=float, default=3.5)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--emit-stride", type=int, default=2)

    parser.add_argument(
        "--cones-layout",
        choices=["24", "4"],
        default="24",
        help="24 = two-cone mode, 4 = four-cone mode",
    )

    parser.add_argument("--refA", help="Reference cone A (four-cone mode)")
    parser.add_argument("--refB", help="Reference cone B (four-cone mode)")

    parser.add_argument(
        "--cone-spacing-vertical-m",
        type=float,
        default=8.0,
        help="Vertical spacing (four-cone mode)",
    )
    parser.add_argument(
        "--cone-spacing-horizontal-m",
        type=float,
        default=6.0,
        help="Horizontal spacing (four-cone mode)",
    )

    args = parser.parse_args()
    asyncio.run(serve_async(args))
