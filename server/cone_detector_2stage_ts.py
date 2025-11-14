# cone_detector_2stage.py
# ============================================================
# 2-Stage pipeline (old-frame reuse + coordinate stability):
#   Stage 1) One-pass video scan → Automatically select a “vertical adjacent 2-cone pair” among 24 cones
#   Stage 2) Using the selected 2 reference cones, estimate drone GPS from pixel (cx, cy) for each frame
#
# Tracking strategy (legacy-style):
#   - User manually draws ROI on the first frame.
#   - Track ROI using OpenCV KCF.
#   - On tracking failure, immediately re-draw ROI on the current frame.
#   - Prevent oversized ROI window: always display scaled preview for ROI selection.
#
# Dependencies: opencv-python, numpy
#   pip install opencv-python numpy
#
# Usage example:
#   out = run_2stage_drag_kcf(
#       video_path="your_video.mp4",
#       cones_gps_json_path="cones_gps.json",
#       cone_spacing_m=4.32,
#       sample_stride=10,
#       min_frames=40,
#       save_csv="drone_gps.csv",
#       visualize=True,
#       display_max=(1280, 720),
#       exit_on_cancel=True,  # ESC cancellation saves results and exits
#   )
#   print(out)
# ============================================================

import cv2
import csv
import math
import json
import numpy as np
from typing import Dict, Tuple, List, Optional

from datetime import datetime, timedelta


# -----------------------------------------------------------
# Timestamp parsing utilities
# -----------------------------------------------------------

def _parse_manual_start(s: Optional[str]) -> Optional[datetime]:
    """Parse manual start timestamp in various formats."""
    if not s:
        return None
    s = s.strip()
    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        if "." in s:
            return datetime.fromtimestamp(float(s))
        return datetime.fromtimestamp(int(s))
    except Exception:
        return None


def _calc_ts_iso(cap, frame_idx: int, fps: float, t0: Optional[datetime]) -> str:
    """Return ISO timestamp (ms precision) using either cap position or fps."""
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms and pos_ms > 0:
        dt = (t0 or datetime.fromtimestamp(0)) + timedelta(milliseconds=float(pos_ms))
    elif fps and fps > 0:
        dt = (t0 or datetime.fromtimestamp(0)) + timedelta(seconds=frame_idx / float(fps))
    else:
        dt = (t0 or datetime.fromtimestamp(0))
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# -----------------------------------------------------------
# Cone detection utilities
# -----------------------------------------------------------

def detect_cones_hsv(frame: np.ndarray) -> List[Tuple[int, int]]:
    """Detect red cones via HSV, return list of (cx, cy)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100], np.uint8)
    upper_red1 = np.array([10, 255, 255], np.uint8)
    lower_red2 = np.array([160, 100, 100], np.uint8)
    upper_red2 = np.array([180, 255, 255], np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pts.append((cx, cy))
    return pts


def remove_outliers_by_mean_distance(points: List[Tuple[int, int]], target_count: int = 4):
    """
    Keep points with smallest average pairwise distance.
    If count ≤ target_count, return unchanged.
    """
    if len(points) <= target_count:
        return points
    arr = np.array(points, float)
    dists = np.linalg.norm(arr[:, None] - arr[None], axis=2)
    avg = np.mean(dists, axis=1)
    keep = np.argsort(avg)[:target_count]
    return [points[i] for i in keep]


# -----------------------------------------------------------
# 24-cone grid indexing
# -----------------------------------------------------------

def cluster_rows_by_y(cone_pixels: List[Tuple[float, float]], n_rows: int = 5):
    """Cluster sorted-by-y cones into 5 rows based on large y-gaps."""
    ys = np.array([p[1] for p in cone_pixels])
    sorted_idx = np.argsort(ys)
    sorted_pixels = [cone_pixels[i] for i in sorted_idx]
    y_vals = [p[1] for p in sorted_pixels]
    if len(y_vals) < n_rows:
        return [sorted_pixels]
    y_diff = np.diff(y_vals)
    split_idx = np.argsort(y_diff)[-(n_rows - 1):] + 1
    split_idx = sorted(split_idx)
    rows = []
    prev = 0
    for idx in split_idx:
        rows.append(sorted_pixels[prev:idx])
        prev = idx
    rows.append(sorted_pixels[prev:])
    return rows


def index_grid_24(cone_pixels: List[Tuple[int, int]]) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Map exactly 24 cones to grid positions (r,c) except center (2,2).
    Return dict key="r_c":(x,y), or None if failed.
    """
    if len(cone_pixels) != 24:
        return None
    rows = cluster_rows_by_y(cone_pixels, 5)
    rows = [sorted(r, key=lambda x: x[0]) for r in rows]
    if len(rows) != 5:
        return None
    mapping = {}
    for r in range(5):
        c_idx = 0
        for c in range(5):
            if (r, c) == (2, 2):
                continue
            try:
                mapping[f"{r}_{c}"] = tuple(map(int, rows[r][c_idx]))
            except Exception:
                return None
            c_idx += 1
    return mapping


# -----------------------------------------------------------
# Stage 1: Automatic 2-cone selection
# -----------------------------------------------------------

class TwoConePreselector:
    """Scan video to find the best vertical adjacent pair among 24 cones."""

    def __init__(self, sample_stride=10, target_count=24,
                 expect_vertical=True, min_frames=40):
        self.sample_stride = sample_stride
        self.target_count = target_count
        self.expect_vertical = expect_vertical
        self.min_frames = min_frames

    def _pair_iter(self):
        """Yield all vertical-adjacent index pairs (r,c)-(r+1,c)."""
        for c in range(5):
            for r in range(4):
                if (r, c) == (1, 2):
                    continue
                if (r + 1, c) == (2, 2):
                    continue
                yield (r, c), (r + 1, c)

    def scan_video(self, video_path, start_time=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        t0 = _parse_manual_start(start_time)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None

        frame_idx = 0
        stats = {}

        while True:
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % self.sample_stride != 0:
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            cones = detect_cones_hsv(frame)
            cones = remove_outliers_by_mean_distance(cones, target_count=self.target_count)
            mapping = index_grid_24(cones)
            if mapping is None:
                continue

            for (r1, c1), (r2, c2) in self._pair_iter():
                k1, k2 = f"{r1}_{c1}", f"{r2}_{c2}"
                if k1 not in mapping or k2 not in mapping:
                    continue
                x1, y1 = mapping[k1]
                x2, y2 = mapping[k2]
                vx, vy = x2 - x1, y2 - y1
                L = math.hypot(vx, vy)
                if L <= 0:
                    continue
                verticality = abs(vx) / L
                pxdist = L
                xmid = 0.5 * (x1 + x2)
                key = (k1, k2)
                bucket = stats.setdefault(key, {"count": [], "v": [], "d": [], "x": []})
                bucket["count"].append(1)
                bucket["v"].append(verticality)
                bucket["d"].append(pxdist)
                bucket["x"].append(xmid)

        cap.release()
        if not stats:
            print("[ERROR] No vertical candidate pairs found.")
            return None

        best_key = None
        best_score = -1e9
        best_summary = {}

        for key, d in stats.items():
            n = len(d["count"])
            if n < self.min_frames:
                continue
            mean_v = float(np.mean(d["v"]))
            std_d = float(np.std(d["d"]))
            std_x = float(np.std(d["x"]))
            score = 5.0 * n - 200.0 * mean_v - 0.5 * std_d - 0.1 * std_x
            if score > best_score:
                best_score = score
                best_key = key
                best_summary = {
                    "frames": n,
                    "verticality_mean": mean_v,
                    "pxdist_median": float(np.median(d["d"])),
                    "x_std": std_x,
                    "score": score,
                }

        if best_key is None:
            print("[ERROR] No pair satisfies min_frames threshold.")
            return None

        return best_key[0], best_key[1], best_summary


# -----------------------------------------------------------
# 2-cone GPS mapping utilities
# -----------------------------------------------------------

class DroneGPSMapper2:
    """Map drone (pixel) → GPS using exactly two reference cones."""

    def __init__(self, cones_gps, cone_spacing_m=3.5):
        self.cones_gps = dict(cones_gps)
        self.cone_spacing_m = float(cone_spacing_m)

        self.unit_east_lat = 0.0
        self.unit_east_lon = 0.0
        self.unit_north_lat = 0.0
        self.unit_north_lon = 0.0

        self.cone_pixel_map = {}
        self.valid = False

    @staticmethod
    def _to_np(p):
        return np.array(p, float)

    def _configure_units_from_two_points(self, latA, lonA, latB, lonB):
        """Configure N/E degree-per-meter scaling using world-space A→B."""
        L = self.cone_spacing_m
        dlat = latB - latA
        dlon = lonB - lonA
        self.unit_north_lat = dlat / L
        self.unit_north_lon = dlon / L
        self.unit_east_lat = -self.unit_north_lon
        self.unit_east_lon = self.unit_north_lat

    @staticmethod
    def _img_axes_and_scale_two_cones(pA, pB, Lw_m):
        """Image N/E axes and pixel→meter scale from two cones."""
        v = pB - pA
        Li = np.linalg.norm(v)
        if Li == 0:
            raise ValueError("Cone pixels identical.")
        eNi = v / Li
        eEi = np.array([-eNi[1], eNi[0]])
        s = Lw_m / Li
        return eEi, eNi, float(s)

    def update_cone_pixel_map(self, cone_pixels):
        """Register two cone pixel coordinates."""
        if isinstance(cone_pixels, dict):
            items = list(cone_pixels.items())
        elif isinstance(cone_pixels, (list, tuple)) and len(cone_pixels) == 2:
            keys = list(self.cones_gps.keys())[:2]
            items = [(keys[0], cone_pixels[0]), (keys[1], cone_pixels[1])]
        else:
            raise ValueError("cone_pixels must be a dict or a 2-tuple")

        if len(items) != 2:
            raise ValueError("Need exactly two cones")

        self.cone_pixel_map = {k: (int(p[0]), int(p[1])) for k, p in items}
        self.valid = True

    def estimate_drone_gps(self, drone_xy):
        """Estimate (lat, lon) from drone pixel (cx, cy)."""
        if not self.valid or len(self.cone_pixel_map) != 2:
            return None

        items = list(self.cone_pixel_map.items())
        (k1, p1), (k2, p2) = items[0], items[1]

        if p1[1] >= p2[1]:
            kA, pA = k1, self._to_np(p1)
            kB, pB = k2, self._to_np(p2)
        else:
            kA, pA = k2, self._to_np(p2)
            kB, pB = k1, self._to_np(p1)

        latA, lonA = self.cones_gps[kA]
        latB, lonB = self.cones_gps[kB]

        self._configure_units_from_two_points(latA, lonA, latB, lonB)
        eEi, eNi, s = self._img_axes_and_scale_two_cones(pA, pB, self.cone_spacing_m)

        pd = self._to_np(drone_xy)
        d_img = pd - pA
        east_px = np.dot(d_img, eEi)
        north_px = np.dot(d_img, eNi)

        dE_m = s * east_px
        dN_m = s * north_px

        dlat = dE_m * self.unit_east_lat + dN_m * self.unit_north_lat
        dlon = dE_m * self.unit_east_lon + dN_m * self.unit_north_lon

        return latA + dlat, lonA + dlon


# -----------------------------------------------------------
# Stage 2: GPS estimator using chosen 2 cones
# -----------------------------------------------------------

class TwoConeGPSEstimator:
    """Stage 2: estimate drone GPS using selected pair of cones."""

    def __init__(self, cones_gps_json_path, keyA, keyB, cone_spacing_m=3.5):
        with open(cones_gps_json_path, "r") as f:
            all_gps = json.load(f)
        if keyA not in all_gps or keyB not in all_gps:
            raise KeyError("Missing keys in cones_gps JSON")
        self.keyA = keyA
        self.keyB = keyB
        self.mapper = DroneGPSMapper2({keyA: all_gps[keyA], keyB: all_gps[keyB]},
                                      cone_spacing_m)

        self.expected_px_dist = None
        self.prev_pxA = None
        self.prev_pxB = None

    def _pick_two_from_detections(self, dets):
        """Pick pixels for keyA/keyB if possible, else fallback to best pair."""
        mapping = index_grid_24(dets) if len(dets) == 24 else None
        if mapping and self.keyA in mapping and self.keyB in mapping:
            pA, pB = mapping[self.keyA], mapping[self.keyB]
            self.expected_px_dist = float(np.linalg.norm(np.array(pB) - np.array(pA)))
            return pA, pB

        if len(dets) < 2:
            return None

        arr = [tuple(map(int, p)) for p in dets]
        best = None
        best_score = -1e9

        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                x1, y1 = arr[i]
                x2, y2 = arr[j]
                vx, vy = x2 - x1, y2 - y1
                L = math.hypot(vx, vy)
                if L <= 0:
                    continue
                verticality = abs(vx) / L

                if y1 > y2:
                    pA, pB = (x1, y1), (x2, y2)
                else:
                    pA, pB = (x2, y2), (x1, y1)

                dist_pen = abs(L - self.expected_px_dist) if self.expected_px_dist else 0
                track_pen = 0
                if self.prev_pxA and self.prev_pxB:
                    track_pen = math.hypot(pA[0] - self.prev_pxA[0], pA[1] - self.prev_pxA[1]) + \
                                math.hypot(pB[0] - self.prev_pxB[0], pB[1] - self.prev_pxB[1])

                score = -200 * verticality - 0.5 * dist_pen - 0.05 * track_pen
                if score > best_score:
                    best_score = score
                    best = (pA, pB)

        if not best:
            return None
        pA, pB = best
        self.expected_px_dist = float(np.linalg.norm(np.array(pB) - np.array(pA)))
        return pA, pB

    def estimate_frame(self, frame, drone_xy):
        """Estimate GPS for each frame."""
        cones = detect_cones_hsv(frame)
        if len(cones) > 24:
            cones = remove_outliers_by_mean_distance(cones, 24)
        picked = self._pick_two_from_detections(cones)
        if not picked:
            return None
        pA, pB = picked
        self.prev_pxA, self.prev_pxB = pA, pB
        self.mapper.update_cone_pixel_map({self.keyA: pA, self.keyB: pB})
        return self.mapper.estimate_drone_gps(drone_xy)


# -----------------------------------------------------------
# Tracker & display utilities
# -----------------------------------------------------------

def _create_kcf():
    """Return OpenCV KCF tracker, legacy or new API."""
    try:
        return cv2.legacy.TrackerKCF_create()
    except Exception:
        return cv2.TrackerKCF_create()


def _resize_for_display(img, max_w=1280, max_h=720):
    """Return (scaled_image, scale_factor)."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        out = cv2.resize(img, (int(w * s), int(h * s)))
    else:
        out = img.copy()
    return out, s


def _select_roi_scaled(img, title="Select", max_w=1280, max_h=720):
    """Scaled ROI selection → return ROI in original coordinates."""
    disp, s = _resize_for_display(img, max_w, max_h)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(title, disp, False, True)
    cv2.destroyWindow(title)
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    inv = 1.0 / s
    return int(x * inv), int(y * inv), int(w * inv), int(h * inv)


# -----------------------------------------------------------
# Main pipeline (drag ROI + KCF + GPS)
# -----------------------------------------------------------

def run_2stage_drag_kcf(
    video_path,
    cones_gps_json_path,
    start_time=None,
    cone_spacing_m=3.5,
    sample_stride=10,
    min_frames=40,
    save_csv=None,
    visualize=True,
    display_max=(1280, 720),
    exit_on_cancel=True,
):
    """
    End-to-end GPS estimation:
      1) Stage 1: Automatic selection of vertical adjacent 2-cone pair.
      2) User draws ROI on the first frame.
      3) Track ROI via KCF.
      4) On tracking failure, immediately re-draw ROI.
      5) Convert ROI center to GPS based on 2 cones.

    Return result dict.
    """
    results = []
    early_exit = False

    selector = TwoConePreselector(sample_stride, min_frames=min_frames)
    picked = selector.scan_video(video_path, start_time)
    if picked is None:
        raise RuntimeError("Failed to select two cones.")
    keyA, keyB, summary = picked

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    t0 = _parse_manual_start(start_time)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame.")

    x, y, w, h = _select_roi_scaled(first, "Select Drone", *display_max)
    if w <= 0 or h <= 0:
        cap.release()
        if exit_on_cancel:
            if save_csv:
                with open(save_csv, "w", newline="") as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow(["timestamp", "latitude", "longitude"])
                    wcsv.writerows(results)
            return {
                "keyA": keyA,
                "keyB": keyB,
                "summary": summary,
                "num_frames_with_gps": len(results),
                "csv": save_csv,
                "early_exit": True,
                "reason": "initial ROI cancelled",
            }
        raise RuntimeError("Empty ROI.")

    tracker = _create_kcf()
    tracker.init(first, (x, y, w, h))
    box = (x, y, w, h)
    estimator = TwoConeGPSEstimator(cones_gps_json_path, keyA, keyB,
                                    cone_spacing_m=cone_spacing_m)

    frame_idx = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        success, box_new = tracker.update(frame)
        if not success:
            disp, _ = _resize_for_display(frame, *display_max)
            cv2.putText(disp, "Tracking failure: ESC=save&exit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("tracking", disp)
            cv2.waitKey(200)

            rx, ry, rw, rh = _select_roi_scaled(frame, "Re-select Drone", *display_max)
            if rw <= 0 or rh <= 0:
                if exit_on_cancel:
                    early_exit = True
                    break
                continue

            tracker = _create_kcf()
            tracker.init(frame, (rx, ry, rw, rh))
            box = (rx, ry, rw, rh)
        else:
            box = tuple(map(int, box_new))

        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2

        latlon = estimator.estimate_frame(frame, (cx, cy))
        if latlon:
            lat, lon = latlon
            ts_iso = _calc_ts_iso(cap, frame_idx, fps, t0)
            results.append((ts_iso, lat, lon))

        if visualize:
            vis = frame.copy()
            disp, s = _resize_for_display(vis, *display_max)
            sx, sy, sw, sh = int(x * s), int(y * s), int(w * s), int(h * s)
            cv2.rectangle(disp, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.circle(disp, (int((x + w / 2) * s), int((y + h / 2) * s)), 3, (0, 0, 255), -1)
            cv2.putText(disp, f"frame:{frame_idx}  [q:quit | r:ROI | e:exit]",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("tracking", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e'):
                early_exit = True
                break
            if key == ord('r'):
                rx, ry, rw, rh = _select_roi_scaled(frame, "Re-select Drone", *display_max)
                if rw > 0 and rh > 0:
                    tracker = _create_kcf()
                    tracker.init(frame, (rx, ry, rw, rh))
                    box = (rx, ry, rw, rh)

    cap.release()
    if visualize:
        try:
            cv2.destroyWindow("tracking")
        except Exception:
            pass

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["timestamp", "latitude", "longitude"])
            wcsv.writerows(results)

    return {
        "keyA": keyA,
        "keyB": keyB,
        "summary": summary,
        "num_frames_with_gps": len(results),
        "csv": save_csv,
        "early_exit": early_exit,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="2-Stage Drone GPS Estimation")
    parser.add_argument("--video", default="test_video.mp4")
    parser.add_argument("--cones-gps", default="cones_gps.json")
    parser.add_argument("--output-csv", default="drone_gps_output.csv")
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--exit-on-cancel", action="store_true")
    args = parser.parse_args()

    result = run_2stage_drag_kcf(
        video_path=args.video,
        cones_gps_json_path=args.cones_gps,
        start_time=args.start_time,
        save_csv=args.output_csv,
        visualize=True,
        exit_on_cancel=args.exit_on_cancel,
    )
    print("2-Stage processing completed:", result)
