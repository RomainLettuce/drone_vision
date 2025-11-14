# cone_detector_simple_2x2_refpair_roiC.py
# ============================================================
# 2x2 (4-cone) assumption with explicit reference pair (refA, refB).
# Supports horizontal or vertical neighboring reference pairs only.
#
# Features:
# - If 4-cone indexing fails, fallback to previous pxA/pxB.
# - If consecutive failures exceed fail_limit (default 10), stop.
# - On tracking failure: mandatory ROI re-selection window:
#     * Enter = confirm
#     * Esc = cancel (if exit_on_cancel=True → save & exit)
#     * C = save current CSV immediately and exit
# - Initial ROI selection supports the same keys/behavior.
# ============================================================

import cv2
import csv
import json
import math
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta

# ---------------------- Time Utilities ----------------------

def _parse_manual_start(s: Optional[str]) -> Optional[datetime]:
    """Parse manual start time into datetime."""
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
        return datetime.fromtimestamp(float(s)) if "." in s else datetime.fromtimestamp(int(s))
    except Exception:
        return None

def _calc_ts_iso(cap, frame_idx: int, fps: float, t0: Optional[datetime]) -> str:
    """Return an ISO-like timestamp string, using POS_MSEC or FPS."""
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms and pos_ms > 0:
        dt = (t0 or datetime.fromtimestamp(0)) + timedelta(milliseconds=float(pos_ms))
    elif fps and fps > 0:
        dt = (t0 or datetime.fromtimestamp(0)) + timedelta(seconds=frame_idx / float(fps))
    else:
        dt = (t0 or datetime.fromtimestamp(0))
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# ---------------------- Red Cone Detector (for faint / small cones) ----------------------

def detect_cones_hsv(
    frame: np.ndarray,
    min_area: int = 100,
    hue_lo_1: int = 0, hue_hi_1: int = 15,
    hue_lo_2: int = 170, hue_hi_2: int = 180,
    sat_lo: int = 40, val_lo: int = 35,
    k_open: int = 3, k_close: int = 5
) -> List[Tuple[int, int]]:
    """
    Detect red cones via HSV thresholding and morphological filtering.
    Returns list of centroid (x,y).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([hue_lo_1, sat_lo, val_lo], np.uint8)
    upper1 = np.array([hue_hi_1, 255, 255], np.uint8)
    lower2 = np.array([hue_lo_2, sat_lo, val_lo], np.uint8)
    upper2 = np.array([hue_hi_2, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    if k_open > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    if k_close > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts: List[Tuple[int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                pts.append((cx, cy))

    print(f"[DEBUG] Detected cone points: {pts}")
    return pts

def remove_outliers_by_mean_distance(points: List[Tuple[int, int]], target_count: int = 4) -> List[Tuple[int, int]]:
    """Remove outliers by mean distance; keep closest 'target_count' points."""
    if len(points) <= target_count:
        return points
    arr = np.array(points, dtype=float)
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
    avg = np.mean(d, axis=1)
    keep = np.argsort(avg)[:target_count]
    return [points[i] for i in keep]

# ---------------------- 2x2 Indexing Utilities ----------------------

def _cluster_rows_by_lr_then_y(pts: List[Tuple[int,int]]) -> List[List[Tuple[int,int]]]:
    """
    Cluster points into 2x2 grid:
    1) Sort by x (left → right)
    2) Inside each column, sort by y (top → bottom)
    Returns rows: [[top-left, top-right], [bottom-left, bottom-right]]
    """
    if len(pts) < 4:
        return [pts]

    A = np.array(pts, dtype=float)
    xs = A[:, 0]
    ys = A[:, 1]

    order_x = np.argsort(xs)
    left_idx = order_x[:2]
    right_idx = order_x[2:]

    left_sorted = left_idx[np.argsort(ys[left_idx])]
    right_sorted = right_idx[np.argsort(ys[right_idx])]

    top_left = tuple(map(int, A[left_sorted[0]]))
    bottom_left = tuple(map(int, A[left_sorted[1]]))
    top_right = tuple(map(int, A[right_sorted[0]]))
    bottom_right = tuple(map(int, A[right_sorted[1]]))

    row_top = [top_left, top_right]
    row_bottom = [bottom_left, bottom_right]

    # Ensure left → right per row
    if row_top[0][0] > row_top[1][0]:
        row_top = [row_top[1], row_top[0]]
    if row_bottom[0][0] > row_bottom[1][0]:
        row_bottom = [row_bottom[1], row_bottom[0]]

    return [row_top, row_bottom]

def index_grid_2x2(pts: List[Tuple[int,int]]) -> Optional[Dict[str, Tuple[int,int]]]:
    """Return {'0_0','0_1','1_0','1_1'} mapping for 4 points, or None."""
    if len(pts) != 4:
        return None

    rows = _cluster_rows_by_lr_then_y(pts)
    if len(rows) != 2 or any(len(r) != 2 for r in rows):
        return None

    return {
        "0_0": rows[0][0],
        "0_1": rows[0][1],
        "1_0": rows[1][0],
        "1_1": rows[1][1],
    }

# ---------------------- Drone GPS Mapper Using Reference Pair ----------------------

class DroneGPSMapper2:
    """
    Map drone pixel coordinates to GPS position using two known cone GPS points.
    """
    def __init__(self, cones_gps: Dict[str, Tuple[float, float]]):
        self.cones_gps = dict(cones_gps)
        self.eEi = np.array([1.0, 0.0])
        self.eNi = np.array([0.0, 1.0])
        self.s_m_per_px = 1.0
        self.unit_east_lat = 0.0
        self.unit_east_lon = 0.0
        self.unit_north_lat = 0.0
        self.unit_north_lon = 0.0
        self.pixel_map: Dict[str, Tuple[int,int]] = {}

    @staticmethod
    def _to_np(p): return np.array(p, dtype=float)

    @staticmethod
    def _img_axes_and_scale(pA, pB, orientation: str, Lw_m: float):
        """Compute image axes eEi, eNi and scale from pixel to meter."""
        v = pB - pA
        Li = float(np.linalg.norm(v))
        if Li == 0:
            raise ValueError("Reference cones are at identical pixel positions.")
        if orientation == "V":
            eNi = v / Li
            eEi = np.array([-eNi[1], eNi[0]])
        else:
            eEi = v / Li
            eNi = np.array([+eEi[1], -eEi[0]])
        s = Lw_m / Li
        return eEi, eNi, float(s)

    def _configure_world_units(self, latA, lonA, latB, lonB, orientation: str, Lw_m: float):
        """Configure east/north conversion per meter."""
        dlat = float(latB - latA)
        dlon = float(lonB - lonA)
        if orientation == "V":
            self.unit_north_lat = dlat / Lw_m
            self.unit_north_lon = dlon / Lw_m
            self.unit_east_lat  = -self.unit_north_lon
            self.unit_east_lon  = +self.unit_north_lat
        else:
            self.unit_east_lat  = dlat / Lw_m
            self.unit_east_lon  = dlon / Lw_m
            self.unit_north_lat = +self.unit_east_lon
            self.unit_north_lon = -self.unit_east_lat

    def update_pair_and_scale(self, keyA, keyB, pxA, pxB, orientation, Lw_m):
        """Update reference pair pixel map, compute axes, scaling, and world units."""
        p1 = self._to_np(pxA)
        p2 = self._to_np(pxB)

        if orientation == "V":  # Reference pair aligned vertically
            if p1[1] >= p2[1]:
                pA, pB = p1, p2
                kA, kB = keyA, keyB
            else:
                pA, pB = p2, p1
                kA, kB = keyB, keyA
        else:  # Horizontal orientation
            if p1[0] <= p2[0]:
                pA, pB = p1, p2
                kA, kB = keyA, keyB
            else:
                pA, pB = p2, p1
                kA, kB = keyB, keyA

        latA, lonA = self.cones_gps[kA]
        latB, lonB = self.cones_gps[kB]

        self.eEi, self.eNi, self.s_m_per_px = self._img_axes_and_scale(pA, pB, orientation, Lw_m)
        self._configure_world_units(latA, lonA, latB, lonB, orientation, Lw_m)

        self.pixel_map = {kA: tuple(map(int, pA)), kB: tuple(map(int, pB))}
        return kA, kB

    def estimate_drone_gps(self, drone_xy: Tuple[int,int], kA: str) -> Tuple[float,float]:
        """Estimate drone GPS coordinates from pixel coordinate relative to reference cone A."""
        if kA not in self.pixel_map:
            raise RuntimeError("Pixel map is not initialized.")
        pA = self._to_np(self.pixel_map[kA])
        pd = self._to_np(drone_xy)

        d_img = pd - pA
        u_east_px = float(np.dot(d_img, self.eEi))
        v_north_px = float(np.dot(d_img, self.eNi))

        dE_m = self.s_m_per_px * u_east_px
        dN_m = self.s_m_per_px * v_north_px

        latA, lonA = self.cones_gps[kA]
        dlat = dE_m * self.unit_east_lat + dN_m * self.unit_north_lat
        dlon = dE_m * self.unit_east_lon + dN_m * self.unit_north_lon

        return (latA + dlat, lonA + dlon)

# ---------------------- 4-Cone Estimator with Robust Fallback ----------------------

class SimpleRefPairEstimator:
    """
    Robust 2x2 (four-cone) estimator:
    - Supports 4 detections, 3 detections (recovery via previous frame),
      or full fallback to previous ref-pair pixel positions.
    - Stops if consecutive failures exceed fail_limit.
    """
    def __init__(
        self,
        cones_gps_json_path: str,
        refA: str, refB: str,
        cone_spacing_vertical_m: float,
        cone_spacing_horizontal_m: float,
        fail_limit: int = 10
    ):
        with open(cones_gps_json_path, "r") as f:
            all_gps: Dict[str, Tuple[float, float]] = json.load(f)

        if refA not in all_gps or refB not in all_gps:
            raise KeyError("refA/refB must exist in cones_gps JSON.")

        self.refA, self.refB = refA, refB
        rA, cA = map(int, refA.split("_"))
        rB, cB = map(int, refB.split("_"))

        if rA == rB and abs(cA - cB) == 1:
            self.orientation = "H"
            self.pair_spacing_m = float(cone_spacing_horizontal_m)
        elif cA == cB and abs(rA - rB) == 1:
            self.orientation = "V"
            self.pair_spacing_m = float(cone_spacing_vertical_m)
        else:
            raise ValueError("refA/refB must be horizontal or vertical neighbors (no diagonals).")

        # All 4 GPS entries required for 2x2 indexing
        self.all_gps = all_gps
        self.all_keys = list(all_gps.keys())
        if len(self.all_keys) != 4:
            raise ValueError("cones_gps JSON must contain exactly 4 keys for 2x2 layout.")

        two_gps = {refA: all_gps[refA], refB: all_gps[refB]}
        self.mapper = DroneGPSMapper2(two_gps)

        self.prev_pxA = None
        self.prev_pxB = None
        self.prev_full_map: Dict[str, Tuple[int,int]] = {}

        self.consec_fail = 0
        self.fail_limit = int(fail_limit)
        self.kA_sorted: Optional[str] = None

    @staticmethod
    def _assign_detected_to_prev(detections, prev_full_map):
        """
        Assign 3 detected points to closest keys from previous frame.
        Missing one is restored from prev_full_map.
        """
        if not prev_full_map or len(detections) != 3:
            return None

        remaining = dict(prev_full_map)
        mapping = {}

        for p in detections:
            px = np.array(p, dtype=float)
            best_k, best_d = None, 1e18
            for k, q in remaining.items():
                d = float(np.hypot(px[0] - q[0], px[1] - q[1]))
                if d < best_d:
                    best_d = d
                    best_k = k
            if best_k is None:
                return None
            mapping[best_k] = tuple(map(int, p))
            remaining.pop(best_k)

        if len(remaining) != 1:
            return None

        missing_key, missing_pt = next(iter(remaining.items()))
        mapping[missing_key] = tuple(map(int, missing_pt))
        return mapping

    def estimate_frame(self, frame, drone_xy):
        """
        Return (latlon | None, used_fallback, consecutive_failures)
        """
        cones = detect_cones_hsv(frame)

        if len(cones) > 4:
            cones = remove_outliers_by_mean_distance(cones, 4)

        used_fallback = False
        mapping = None

        # 4 detections → standard indexing
        if len(cones) == 4:
            mapping = index_grid_2x2(cones)
            if mapping and self.refA in mapping and self.refB in mapping:
                self.prev_full_map = dict(mapping)
                self.prev_pxA = mapping[self.refA]
                self.prev_pxB = mapping[self.refB]
                self.consec_fail = 0
            else:
                mapping = None

        # 3 detections → recover from previous frame
        elif len(cones) == 3 and self.prev_full_map:
            mapping = self._assign_detected_to_prev(cones, self.prev_full_map)
            if mapping and self.refA in mapping and self.refB in mapping:
                self.prev_full_map = dict(mapping)
                self.prev_pxA = mapping[self.refA]
                self.prev_pxB = mapping[self.refB]
                self.consec_fail = 0
                used_fallback = True
            else:
                mapping = None

        # If mapping succeeded
        if mapping and self.refA in mapping and self.refB in mapping:
            pxA = mapping[self.refA]
            pxB = mapping[self.refB]
            kA_sorted, _ = self.mapper.update_pair_and_scale(
                self.refA, self.refB, pxA, pxB,
                self.orientation, self.pair_spacing_m
            )
            self.kA_sorted = kA_sorted
            latlon = self.mapper.estimate_drone_gps(drone_xy, kA_sorted)
            return latlon, used_fallback, self.consec_fail

        # If mapping failed → fallback
        self.consec_fail += 1
        used_fallback = True

        if self.prev_pxA is None or self.prev_pxB is None:
            return None, used_fallback, self.consec_fail

        # Use previous ref-pair
        pxA, pxB = self.prev_pxA, self.prev_pxB
        kA_sorted, _ = self.mapper.update_pair_and_scale(
            self.refA, self.refB, pxA, pxB,
            self.orientation, self.pair_spacing_m
        )
        self.kA_sorted = kA_sorted
        latlon = self.mapper.estimate_drone_gps(drone_xy, kA_sorted)
        return latlon, used_fallback, self.consec_fail

# ---------------------- ROI Selection Window ----------------------

def _resize_for_display(img, max_w=1280, max_h=720):
    """Resize image to fit within max_w x max_h while preserving aspect ratio."""
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h), 1.0)
    if s < 1.0:
        disp = cv2.resize(img, (int(w*s), int(h*s)))
    else:
        disp = img.copy()
    return disp, s

def _select_roi_scaled_custom(img, title="Select ROI", max_w=1280, max_h=720):
    """
    Custom ROI selection window.
    Keys:
      - Enter: confirm ROI
      - Esc: cancel
      - C: save & exit
      - R: reset selection
    Returns: (x,y,w,h,status) where status ∈ {"ok","cancel","save_exit"}.
    """
    disp, s = _resize_for_display(img, max_w, max_h)
    clone = disp.copy()
    roi = [0,0,0,0]
    drawing = [False]
    pt1 = [0,0]

    help_text = "Drag to draw ROI | Enter: OK | C: Save&Exit | Esc: Cancel | R: Reset"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, max_w, max_h)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            pt1[0], pt1[1] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            img2 = clone.copy()
            x0,y0 = pt1
            cv2.rectangle(img2, (x0,y0), (x,y), (0,255,0), 2)
            cv2.putText(img2, help_text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
            cv2.imshow(title, img2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            x0,y0 = pt1
            x_min,y_min = min(x0,x), min(y0,y)
            x_max,y_max = max(x0,x), max(y0,y)
            roi[:] = [x_min, y_min, x_max - x_min, y_max - y_min]
            img2 = clone.copy()
            cv2.rectangle(img2, (x_min,y_min),(x_max,y_max),(0,255,0),2)
            cv2.putText(img2, help_text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
            cv2.imshow(title, img2)

    cv2.setMouseCallback(title, on_mouse)
    cv2.putText(disp, help_text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
    cv2.imshow(title, disp)

    status = "cancel"

    while True:
        k = cv2.waitKey(10) & 0xFF
        if k in (13,10):  # Enter
            if roi[2] > 0 and roi[3] > 0:
                status = "ok"
                break
        elif k == 27:  # Esc
            status = "cancel"
            break
        elif k in (ord('c'), ord('C')):
            status = "save_exit"
            break
        elif k in (ord('r'), ord('R')):
            roi = [0,0,0,0]
            disp = clone.copy()
            cv2.putText(disp, help_text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
            cv2.imshow(title, disp)

    cv2.destroyWindow(title)

    inv = 1.0 / s
    x,y,w,h = [int(v*inv) for v in roi]
    return x,y,w,h,status

# ---------------------- Tracker Generator ----------------------

def _create_kcf():
    """Create a KCF tracker with legacy fallback."""
    try:
        return cv2.legacy.TrackerKCF_create()
    except Exception:
        return cv2.TrackerKCF_create()

# ---------------------- End-to-End Main Pipeline ----------------------

def run_2stage_drag_kcf(
    video_path: str,
    cones_gps_json_path: str,
    refA: str, refB: str,
    start_time: Optional[str] = None,
    cone_spacing_vertical_m: float = 8.0,
    cone_spacing_horizontal_m: float = 6.0,
    save_csv: Optional[str] = None,
    visualize: bool = True,
    display_max: Tuple[int, int] = (1280,720),
    exit_on_cancel: bool = True,
    fail_limit: int = 10,
    reselect_display_max: Tuple[int, int] = (1920,1080),
):
    """
    Full end-to-end drone tracking + GPS estimation with:
    - initial ROI selection
    - KCF tracking
    - 4-cone robust mapping
    - ROI re-selection on tracking failure
    """
    results = []
    early_exit = False
    fail_reason = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    t0 = _parse_manual_start(start_time)

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame.")

    # ---------------- Initial ROI ----------------
    print("[INFO] Waiting for initial ROI selection...")
    x,y,w,h,status = _select_roi_scaled_custom(first, "Select Drone", *display_max)

    if status == "save_exit":
        cap.release()
        _save_csv(save_csv, results)
        return {"early_exit": True, "reason": "save&exit at initial ROI", "csv": save_csv}

    if status == "cancel" or w <= 0 or h <= 0:
        cap.release()
        if exit_on_cancel:
            _save_csv(save_csv, results)
            return {"early_exit": True, "reason": "initial ROI canceled", "csv": save_csv}
        else:
            raise RuntimeError("Initial ROI selection canceled.")

    tracker = _create_kcf()
    tracker.init(first, (x,y,w,h))
    box = (x,y,w,h)

    estimator = SimpleRefPairEstimator(
        cones_gps_json_path, refA, refB,
        cone_spacing_vertical_m, cone_spacing_horizontal_m,
        fail_limit=fail_limit
    )

    frame_idx = 1

    # ---------------- Main Loop ----------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        success, box_new = tracker.update(frame)
        if not success:
            # ---------------- Tracking Failed → Re-select ROI ----------------
            print("[WARN] Tracking failed → opening ROI re-selection window...")
            disp, _ = _resize_for_display(frame, *reselect_display_max)
            cv2.putText(disp, "Tracking failed → Re-select ROI (Enter OK / C Save&Exit / Esc Cancel)",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)
            cv2.imshow("tracking", disp)
            cv2.waitKey(200)

            rx, ry, rw, rh, rstat = _select_roi_scaled_custom(
                frame, "Re-select Drone", *reselect_display_max
            )

            if rstat == "save_exit":
                early_exit = True; fail_reason = "save&exit at reselect"
                break
            elif rstat == "cancel" or rw <= 0 or rh <= 0:
                if exit_on_cancel:
                    early_exit = True; fail_reason = "ROI re-selection canceled"
                    break
                # Continue using previous ROI (unsafe but allowed)
            else:
                tracker = _create_kcf()
                tracker.init(frame, (rx,ry,rw,rh))
                x,y,w,h = rx,ry,rw,rh
                box = (x,y,w,h)
        else:
            box = tuple(map(int, box_new))

        x,y,w,h = box
        cx, cy = x + w//2, y + h//2

        latlon, used_fallback, fail_streak = estimator.estimate_frame(frame, (cx,cy))
        if latlon is not None:
            lat, lon = latlon
            ts_iso = _calc_ts_iso(cap, frame_idx, fps, t0)
            results.append((ts_iso, lat, lon))

        # ---------------- Visualization ----------------
        if visualize:
            vis = frame.copy()
            disp, s = _resize_for_display(vis, *display_max)
            sx, sy, sw, sh = int(x*s), int(y*s), int(w*s), int(h*s)
            cv2.rectangle(disp, (sx,sy), (sx+sw,sy+sh), (0,255,0), 2)
            cv2.circle(disp, (int((x+w/2)*s), int((y+h/2)*s)), 3, (0,0,255), -1)

            msg = f"Frame:{frame_idx}  Fail:{fail_streak}  [{'FB' if used_fallback else 'OK'}]"
            cv2.putText(disp, msg, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
            cv2.imshow("tracking", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                early_exit = True; fail_reason = "quit by user"; break
            if key == ord('e'):
                early_exit = True; fail_reason = "save&exit by user"; break
            if key == ord('r'):
                print("[INFO] Manual ROI re-selection requested.")
                rx, ry, rw, rh, rstat = _select_roi_scaled_custom(
                    frame, "Re-select Drone", *display_max
                )
                if rstat == "save_exit":
                    early_exit = True; fail_reason = "save&exit at manual reselect"
                    break
                if rstat == "ok" and rw > 0 and rh > 0:
                    tracker = _create_kcf()
                    tracker.init(frame, (rx,ry,rw,rh))
                    x,y,w,h = rx,ry,rw,rh
                    box = (x,y,w,h)
                elif rstat == "cancel" and exit_on_cancel:
                    early_exit = True; fail_reason = "manual reselect canceled"
                    break

        # Stop if too many consecutive indexing failures
        if fail_streak >= estimator.fail_limit:
            early_exit = True
            fail_reason = f"Cone indexing failed {fail_streak} consecutive frames"
            print(f"[ERROR] {fail_reason}")
            break

    cap.release()
    try: cv2.destroyWindow("tracking")
    except: pass

    _save_csv(save_csv, results)

    return {
        "refA": refA,
        "refB": refB,
        "orientation": getattr(estimator, "orientation", None),
        "num_frames_with_gps": len(results),
        "csv": save_csv,
        "early_exit": early_exit,
        **({"reason": fail_reason} if early_exit and fail_reason else {}),
    }

def _save_csv(path, rows):
    """Save GPS results to CSV."""
    if not path:
        return
    with open(path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["timestamp","latitude","longitude"])
        wcsv.writerows(rows)

# ---------------------- CLI ----------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple 2x2 cones GPS estimator with ROI reselect")

    parser.add_argument("--video", required=True)
    parser.add_argument("--cones-gps", required=True)
    parser.add_argument("--output-csv", required=True)

    parser.add_argument("--start-time", default=None)
    parser.add_argument("--exit-on-cancel", action="store_true")

    parser.add_argument("--cone-spacing-vertical-m", type=float, default=8.0)
    parser.add_argument("--cone-spacing-horizontal-m", type=float, default=6.0)

    parser.add_argument("--refA", required=True)
    parser.add_argument("--refB", required=True)

    parser.add_argument("--fail-limit", type=int, default=10)

    args = parser.parse_args()

    out = run_2stage_drag_kcf(
        video_path=args.video,
        cones_gps_json_path=args.cones_gps,
        refA=args.refA,
        refB=args.refB,
        start_time=args.start_time,
        cone_spacing_vertical_m=args.cone_spacing_vertical_m,
        cone_spacing_horizontal_m=args.cone_spacing_horizontal_m,
        save_csv=args.output_csv,
        visualize=True,
        exit_on_cancel=args.exit_on_cancel,
        fail_limit=args.fail_limit,
    )

    print("Completed:", out)
