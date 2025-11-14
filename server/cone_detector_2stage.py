# cone_detector_2stage.py
# ============================================================
# 2-Stage 파이프라인 (옛날 영상 재활용 + 좌표 튐 방지):
#   Stage 1) 영상 1패스 스캔 → "수직 인접 2콘" 자동 선택 (24콘 인덱싱/아웃라이어 제거)
#   Stage 2) 선택한 2콘을 기준으로, 매 프레임 드론 픽셀(cx,cy) → GPS 추정
#
# 트래킹 방식: "원래처럼"
#   - 첫 프레임에서 사용자가 직접 드래그로 ROI 지정
#   - OpenCV KCF 트래커로 추적
#   - 추적 실패 시, 그 프레임에서 즉시 다시 드래그
#   - 화면 너무 커지는 문제 방지: 항상 축소 프리뷰에서 ROI 선택/표시
#
# 필요 패키지: opencv-python, numpy
#   pip install opencv-python numpy
#
# 사용 예시:
#   out = run_2stage_drag_kcf(
#       video_path="your_video.mp4",
#       cones_gps_json_path="cones_gps.json",  # "r_c" 키로 24콘 GPS가 들어있는 JSON
#       cone_spacing_m=4.32,
#       sample_stride=10,
#       min_frames=40,
#       save_csv="drone_gps.csv",
#       visualize=True,
#       display_max=(1280, 720),
#       exit_on_cancel=True,  # ★ ESC 취소 시 지금까지 결과 저장 후 종료
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

def _parse_manual_start(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    # Try ISO
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
    # Epoch seconds
    try:
        if "." in s:
            return datetime.fromtimestamp(float(s))
        return datetime.fromtimestamp(int(s))
    except Exception:
        return None

def _calc_ts_iso(cap, frame_idx: int, fps: float, t0: Optional[datetime]) -> str:
    # Prefer cap position in ms if available; fallback to fps
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms and pos_ms > 0:
        dt = (t0 or datetime.fromtimestamp(0)) + timedelta(milliseconds=float(pos_ms))
    elif fps and fps > 0:
        dt = (t0 or datetime.fromtimestamp(0)) + timedelta(seconds=frame_idx / float(fps))
    else:
        dt = (t0 or datetime.fromtimestamp(0))
    # ISO to millisecond precision
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ---------------------- 콘 검출 & 전처리 ----------------------

def detect_cones_hsv(frame: np.ndarray) -> List[Tuple[int, int]]:
    viz_frame = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨강 두 구간
    lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 100, 100], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts: List[Tuple[int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pts.append((cx, cy))
                cv2.circle(viz_frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(viz_frame, f"Cone: ({cx}, {cy})", (cx-10, cy-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return pts


def remove_outliers_by_mean_distance(points: List[Tuple[int, int]], target_count: int = 4) -> List[Tuple[int, int]]:
    """
    평균 거리 기반으로 outlier를 제거하여 target_count개 남긴다.
    (검출점이 target_count보다 많을 때만 적용; 적으면 그대로 반환)
    """
    if len(points) <= target_count:
        return points
    arr = np.array(points, dtype=float)
    dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
    avg = np.mean(dists, axis=1)
    keep_idx = np.argsort(avg)[:target_count]
    return [points[i] for i in keep_idx]


# ---------------------- 그리드 인덱싱 (24개) ----------------------

def cluster_rows_by_y(cone_pixels: List[Tuple[float, float]], n_rows: int = 5):
    """
    y 좌표로 정렬해 큰 간격 기준으로 5개 행으로 분할.
    (5x5 그리드에서 중심(2,2) 제외 = 24개를 (r,c)로 매핑하는데 사용)
    """
    ys = np.array([p[1] for p in cone_pixels])
    sorted_indices = np.argsort(ys)
    sorted_pixels = [cone_pixels[i] for i in sorted_indices]
    y_values = [p[1] for p in sorted_pixels]
    if len(y_values) < n_rows:
        return [sorted_pixels]
    y_diffs = np.diff(y_values)
    split_indices = np.argsort(y_diffs)[-(n_rows - 1):] + 1
    split_indices = sorted(split_indices)
    row_splits: List[List[Tuple[float, float]]] = []
    prev = 0
    for idx in split_indices:
        row_splits.append(sorted_pixels[prev:idx])
        prev = idx
    row_splits.append(sorted_pixels[prev:])
    return row_splits


def index_grid_24(cone_pixels: List[Tuple[int, int]]) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    24개 콘을 (r,c) 키로 매핑. 실패 시 None.
    키: "r_c"  (r=0..4, c=0..4, 단 (2,2) 제외)
    """
    if len(cone_pixels) != 24:
        return None
    rows = cluster_rows_by_y(cone_pixels, n_rows=5)
    rows = [sorted(row, key=lambda p: p[0]) for row in rows]
    if len(rows) != 5:
        return None
    mapping: Dict[str, Tuple[int, int]] = {}
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


# ---------------------- Stage 1: 2콘 자동 선택 ----------------------

class TwoConePreselector:
    """
    영상 1패스 스캔으로 "수직 인접 2콘"을 고르는 클래스.
    - 등장 프레임 수가 많고
    - 수직성(가로 성분/길이)이 낮고(=더 수직)
    - 거리/열 위치 분산이 낮은 쌍을 선택
    """

    def __init__(self, sample_stride: int = 10, target_count: int = 24,
                 expect_vertical: bool = True, min_frames: int = 40):
        self.sample_stride = sample_stride
        self.target_count = target_count
        self.expect_vertical = expect_vertical
        self.min_frames = min_frames

    def _pair_iter(self):
        """
        수직 인접 쌍 (r,c)-(r+1,c) 생성. 중앙 (2,2) 비어있으므로 건너뜀.
        """
        for c in range(5):
            for r in range(4):  # r=0..3 -> (r, r+1)
                if (r, c) == (1, 2):      # 위가 (2,2)인 케이스 방지
                    continue
                if (r + 1, c) == (2, 2):  # 아래가 (2,2)인 케이스 방지
                    continue
                yield (r, c), (r + 1, c)

    def scan_video(self, video_path, start_time: Optional[str] = None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        t0 = _parse_manual_start(start_time)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None

        frame_idx = 0
        stats: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

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

            # 후보 수직 인접 쌍 모두 평가
            for (r1, c1), (r2, c2) in self._pair_iter():
                k1, k2 = f"{r1}_{c1}", f"{r2}_{c2}"
                if k1 not in mapping or k2 not in mapping:
                    continue
                x1, y1 = mapping[k1]
                x2, y2 = mapping[k2]
                vx, vy = (x2 - x1), (y2 - y1)
                L = math.hypot(vx, vy)
                if L <= 0:
                    continue

                # 수직도: |vx|/L (0이면 완전 수직)
                verticality = abs(vx) / L
                pxdist = L
                xmid = 0.5 * (x1 + x2)

                key = (k1, k2)
                bucket = stats.setdefault(key, {"count": [], "v": [], "d": [], "x": []})
                bucket["count"].append(1.0)
                bucket["v"].append(verticality)
                bucket["d"].append(pxdist)
                bucket["x"].append(xmid)

        cap.release()
        if not stats:
            print("[ERROR] No stable pairs found.")
            return None

        # 점수 계산: 많이 보일수록↑, 수직성 낮을수록(=더 수직)↑, 거리/위치 분산 낮을수록↑
        best_key = None
        best_score = -1e9
        best_summary: Dict[str, float] = {}

        for key, d in stats.items():
            n = len(d["count"])
            mean_v = float(np.mean(d["v"])) if d["v"] else 1.0
            std_d = float(np.std(d["d"])) if d["d"] else 1e3
            std_x = float(np.std(d["x"])) if d["x"] else 1e3
            score = 5.0 * n - 200.0 * mean_v - 0.5 * std_d - 0.1 * std_x
            if n >= self.min_frames and score > best_score:
                best_score = score
                best_key = key
                best_summary = {
                    "frames": n,
                    "verticality_mean": mean_v,
                    "pxdist_median": float(np.median(d["d"])) if d["d"] else 0.0,
                    "x_std": std_x,
                    "score": score,
                }

        if best_key is None:
            print("[ERROR] No pair passed the min_frames threshold.")
            return None

        kA, kB = best_key  # 아래(r), 위(r+1)
        return kA, kB, best_summary


# ---------------------- DroneGPSMapper (2콘 전용, 자급자족) ----------------------

class DroneGPSMapper2:
    """
    2개 콘 전용 GPS 매퍼 (외부 파일 의존 없이 이 안에서 동작)
    - update_cone_pixel_map({keyA: (xA, yA), keyB: (xB, yB)})
      * 픽셀 y 기준으로 아래(A)=큰 y, 위(B)=작은 y로 자동 정렬
    - estimate_drone_gps((xd, yd)) -> (lat, lon)
      * 콘 간 실제 거리 cone_spacing_m (기본 4.32m)
      * 세계축: 콘A→콘B를 북(North), 동(East)=시계 90°
      * 이미지축: pA→pB를 이미지 북, 직교를 이미지 동 + m/px 스케일
    """

    def __init__(self, cones_gps: Dict[str, Tuple[float, float]], cone_spacing_m: float = 3.5):
        if len(cones_gps) < 2:
            raise ValueError("cones_gps must contain at least 2 entries")
        self.cones_gps = dict(cones_gps)  # key -> (lat, lon)
        self.cone_spacing_m = float(cone_spacing_m)

        # 단위변환 계수(미터→도)
        self.unit_east_lat = 0.0
        self.unit_east_lon = 0.0
        self.unit_north_lat = 0.0
        self.unit_north_lon = 0.0

        # 픽셀 맵 (2개)
        self.cone_pixel_map: Dict[str, Tuple[int, int]] = {}
        self.valid = False

    # ---- 내부 유틸 ----
    @staticmethod
    def _to_np(p):
        return np.array(p, dtype=float)

    def _configure_units_from_two_points(self, latA: float, lonA: float, latB: float, lonB: float):
        """
        A=남/아래, B=북/위. A→B를 북(North)으로 두고, 동(East)은 시계방향 90° 회전.
        unit_*: [미터] → [도] 변환 계수.
        """
        L = self.cone_spacing_m
        dlat = float(latB - latA)
        dlon = float(lonB - lonA)
        # 북(미터→도)
        self.unit_north_lat = dlat / L
        self.unit_north_lon = dlon / L
        # 동(시계 90° 회전)
        self.unit_east_lat = -self.unit_north_lon
        self.unit_east_lon = +self.unit_north_lat

    @staticmethod
    def _img_axes_and_scale_two_cones(pA: np.ndarray, pB: np.ndarray, Lw_m: float):
        """
        이미지 축 및 스케일 (m/px):
        pA(아래) → pB(위) 방향 = 이미지 북(N), 동(E)=우회전 90°
        """
        v = pB - pA
        Li = float(np.linalg.norm(v))
        if Li == 0:
            raise ValueError("Two cone pixels are identical.")
        eNi = v / Li
        eEi = np.array([-eNi[1], eNi[0]])  # 90° 회전
        s = Lw_m / Li  # [m/px]
        return eEi, eNi, float(s)

    def update_cone_pixel_map(self, cone_pixels):
        """
        cone_pixels: dict 형태 권장. {keyA:(xA,yA), keyB:(xB,yB)}
        - key 이름은 self.cones_gps에 있는 두 키 중 하나여야 함.
        - 내부에서 y값으로 아래/위(A/B) 판별하여 축 설정용으로 사용.
        """
        if isinstance(cone_pixels, dict):
            items = list(cone_pixels.items())
        elif isinstance(cone_pixels, (list, tuple)) and len(cone_pixels) == 2:
            # 키 정보를 잃으므로, 여긴 dict 사용을 권장. (임시로 key 이름을 첫 두 개로 할당 시도)
            keys = list(self.cones_gps.keys())[:2]
            items = [(keys[0], cone_pixels[0]), (keys[1], cone_pixels[1])]
        else:
            raise ValueError("cone_pixels must be dict with 2 items, or list/tuple of length 2.")

        if len(items) != 2:
            raise ValueError("Need exactly 2 cones.")

        self.cone_pixel_map = {k: (int(p[0]), int(p[1])) for k, p in items}
        self.valid = True

    def estimate_drone_gps(self, drone_xy: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        드론 픽셀 좌표(drone_xy) → GPS (lat, lon)
        - update_cone_pixel_map가 선행되어 있어야 함.
        """
        if not self.valid or len(self.cone_pixel_map) != 2:
            return None

        # 픽셀에서 아래/위 콘 정렬
        items = list(self.cone_pixel_map.items())
        (k1, p1), (k2, p2) = items[0], items[1]
        # y 큰 쪽이 아래(A), 작은 쪽이 위(B)
        if p1[1] >= p2[1]:
            kA, pA = k1, self._to_np(p1)
            kB, pB = k2, self._to_np(p2)
        else:
            kA, pA = k2, self._to_np(p2)
            kB, pB = k1, self._to_np(p1)

        # A/B의 GPS (세계 좌표) 추출
        latA, lonA = self.cones_gps[kA]
        latB, lonB = self.cones_gps[kB]

        # 세계 단위(미터→도): A→B = 북
        self._configure_units_from_two_points(latA, lonA, latB, lonB)

        # 이미지 축/스케일: A→B = 이미지 북
        eEi, eNi, s = self._img_axes_and_scale_two_cones(pA, pB, self.cone_spacing_m)

        # 드론 픽셀 → A 기준 로컬 (동/북 성분) [px]
        pd = self._to_np(drone_xy)
        d_img = pd - pA
        u_east_px = float(np.dot(d_img, eEi))
        v_north_px = float(np.dot(d_img, eNi))

        # [px] → [m]
        dE_m = s * u_east_px
        dN_m = s * v_north_px

        # [m] → [deg]
        dlat = dE_m * self.unit_east_lat + dN_m * self.unit_north_lat
        dlon = dE_m * self.unit_east_lon + dN_m * self.unit_north_lon

        return (latA + dlat, lonA + dlon)


# ---------------------- Stage 2: 2콘 기준 GPS 추정 ----------------------

class TwoConeGPSEstimator:
    """
    Stage 1에서 고른 두 콘을 기준(reference)으로 매 프레임 GPS 산출.
    - 2콘이 그리드 인덱싱으로 바로 매칭되면 그 키 사용
    - 실패 시, 모든 쌍 중 수직성↑ + 거리근사 + 이전 위치 근접성으로 폴백
    """

    def __init__(self, cones_gps_json_path: str, keyA: str, keyB: str, cone_spacing_m: float = 3.5):
        with open(cones_gps_json_path, 'r') as f:
            all_gps: Dict[str, Tuple[float, float]] = json.load(f)
        if keyA not in all_gps or keyB not in all_gps:
            raise KeyError("Selected keys not present in cones_gps JSON")
        two_gps = {keyA: all_gps[keyA], keyB: all_gps[keyB]}

        self.keyA, self.keyB = keyA, keyB
        self.mapper = DroneGPSMapper2(two_gps, cone_spacing_m=cone_spacing_m)
        self.expected_px_dist: Optional[float] = None
        self.prev_pxA: Optional[Tuple[float, float]] = None
        self.prev_pxB: Optional[Tuple[float, float]] = None

    def _pick_two_from_detections(self, detections: List[Tuple[int, int]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        그리드 인덱싱 성공 시 두 키 픽셀 반환.
        실패 시 수직/거리/근접성 기준으로 폴백 탐색.
        """
        mapping = index_grid_24(detections) if len(detections) == 24 else None
        if mapping is not None and self.keyA in mapping and self.keyB in mapping:
            pA, pB = mapping[self.keyA], mapping[self.keyB]
            self.expected_px_dist = float(np.linalg.norm(np.array(pB) - np.array(pA)))
            return pA, pB

        # 폴백: 모든 쌍 중 수직성↑ + 거리 근사 + 이전 위치 근접
        if len(detections) < 2:
            return None
        arr = [tuple(map(int, p)) for p in detections]
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
                # 위/아래 정렬: y 큰 쪽 = 아래(A), y 작은 쪽 = 위(B)
                if y1 > y2:
                    pA, pB = (x1, y1), (x2, y2)
                else:
                    pA, pB = (x2, y2), (x1, y1)

                dist_pen = 0.0
                if self.expected_px_dist is not None:
                    dist_pen = abs(L - self.expected_px_dist)
                track_pen = 0.0
                if self.prev_pxA is not None and self.prev_pxB is not None:
                    track_pen = math.hypot(pA[0] - self.prev_pxA[0], pA[1] - self.prev_pxA[1]) \
                              + math.hypot(pB[0] - self.prev_pxB[0], pB[1] - self.prev_pxB[1])

                score = -200.0 * verticality - 0.5 * dist_pen - 0.05 * track_pen
                if score > best_score:
                    best_score = score
                    best = (pA, pB)

        if best is not None:
            pA, pB = best
            self.expected_px_dist = float(np.linalg.norm(np.array(pB) - np.array(pA)))
            return pA, pB
        return None

    def estimate_frame(self, frame: np.ndarray, drone_xy: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        cones = detect_cones_hsv(frame)
        # 검출점이 24보다 많을 때만 24개로 줄임
        if len(cones) > 24:
            cones = remove_outliers_by_mean_distance(cones, target_count=24)
        picked = self._pick_two_from_detections(cones)
        if picked is None:
            return None
        pA, pB = picked
        self.prev_pxA, self.prev_pxB = pA, pB

        # 2-콘 업데이트 → GPS 추정
        self.mapper.update_cone_pixel_map({self.keyA: pA, self.keyB: pB})
        latlon = self.mapper.estimate_drone_gps(drone_xy)
        return latlon


# ---------------------- 트래커/디스플레이 (원래 방식) ----------------------

def _create_kcf():
    """OpenCV 버전별 legacy/non-legacy 호환."""
    try:
        return cv2.legacy.TrackerKCF_create()
    except Exception:
        return cv2.TrackerKCF_create()

def _resize_for_display(img, max_w=1280, max_h=720):
    """
    화면에 너무 크게 뜨는 문제를 막기 위해, 표시용으로만 축소.
    반환: (표시용 이미지, 스케일값 s).  원본 좌표 = 표시좌표 * (1/s)
    """
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h), 1.0)
    if s < 1.0:
        disp = cv2.resize(img, (int(w * s), int(h * s)))
    else:
        disp = img.copy()
    return disp, s

def _select_roi_scaled(img, title="Select", max_w=1280, max_h=720):
    """
    축소된 프리뷰에서 ROI를 선택하고, 원본 좌표계로 환산해서 반환.
    - ESC로 취소하면 (0,0,0,0) 반환
    """
    disp, s = _resize_for_display(img, max_w, max_h)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(title, disp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    inv = 1.0 / s
    return int(x * inv), int(y * inv), int(w * inv), int(h * inv)


# ---------------------- End-to-End: 드래그+KCF 실행 ----------------------

def run_2stage_drag_kcf(
    video_path: str,
    cones_gps_json_path: str,
    start_time: Optional[str] = None,
    cone_spacing_m: float = 3.5,
    sample_stride: int = 10,
    min_frames: int = 40,
    save_csv: Optional[str] = None,
    visualize: bool = True,
    display_max: Tuple[int, int] = (1280, 720),
    exit_on_cancel: bool = True,     # ★ ESC 취소 시 지금까지 결과 저장 후 종료
) -> Dict[str, object]:
    """
    1) 영상 1패스 스캔으로 '수직 인접 2콘' 자동 선택(robust 인덱싱)
    2) 첫 프레임에서 사용자가 드래그로 ROI 지정 → KCF로 추적
    3) 추적 실패 시, 현재 프레임에서 즉시 재드래그(selectROI)
       - ESC(취소) 시 exit_on_cancel=True면 지금까지 결과 저장 후 종료
    4) 매 프레임 ROI 중심(cx,cy)로 GPS 추정 (2콘 기준)
    """
    results: List[Tuple[str, float, float]] = []  # (timestamp_iso, lat, lon)
    early_exit = False

    # Stage 1: 2콘 자동 선택
    selector = TwoConePreselector(sample_stride=sample_stride, min_frames=min_frames)
    picked = selector.scan_video(video_path, start_time=start_time)
    if picked is None and hasattr(selector, "last_best") and selector.last_best:
        picked = selector.last_best
    if picked is None:
        raise RuntimeError("Failed to preselect two cones.")
    keyA, keyB, summary = picked

    # Stage 2 준비
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    t0 = _parse_manual_start(start_time)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError("Cannot read first frame.")

    # 초기 ROI: 축소 프리뷰에서 드래그 → 원본 좌표로 환산
    x, y, w, h = _select_roi_scaled(first, "Select Drone", *display_max)
    if w <= 0 or h <= 0:
        cap.release()
        if exit_on_cancel:
            # 지금까지 결과 저장 후 깔끔 종료
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
                "reason": "initial ROI canceled",
            }
        else:
            raise RuntimeError("Empty ROI — selection cancelled.")

    tracker = _create_kcf()
    tracker.init(first, (x, y, w, h))
    box = (x, y, w, h)

    estimator = TwoConeGPSEstimator(cones_gps_json_path, keyA, keyB, cone_spacing_m=cone_spacing_m)

    frame_idx = 1  # first는 이미 읽음

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        success, box_new = tracker.update(frame)
        if not success:
            # 실패 시 즉시 재드래그 (ESC=저장 후 종료 안내)
            disp, _ = _resize_for_display(frame, *display_max)
            cv2.putText(disp, "Tracking failure. ESC in ROI window = save & exit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("tracking", disp)
            cv2.waitKey(200)

            x, y, w, h = _select_roi_scaled(frame, "Re-select Drone", *display_max)
            if w <= 0 or h <= 0:
                if exit_on_cancel:
                    early_exit = True
                    break        # 지금까지 결과만 저장하고 종료
                else:
                    # 이전 동작 유지: 스킵하고 계속
                    if visualize:
                        disp, _ = _resize_for_display(frame, *display_max)
                        cv2.imshow("tracking", disp)
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            break
                    continue

            tracker = _create_kcf()
            tracker.init(frame, (x, y, w, h))
            box = (x, y, w, h)
        else:
            box = tuple(map(int, box_new))

        # 중심 좌표
        x, y, w, h = map(int, box)
        cx, cy = x + w // 2, y + h // 2

        # GPS 추정 (2콘)
        latlon = estimator.estimate_frame(frame, (cx, cy))
        if latlon is not None:
            lat, lon = latlon
            ts_iso = _calc_ts_iso(cap, frame_idx, fps, t0)
            results.append((ts_iso, lat, lon))

        # 시각화(축소 프리뷰) + 키 핸들러
        if visualize:
            vis = frame.copy()
            disp, s = _resize_for_display(vis, *display_max)
            sx, sy, sw, sh = int(x * s), int(y * s), int(w * s), int(h * s)
            cv2.rectangle(disp, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.circle(disp, (int((x + w / 2) * s), int((y + h / 2) * s)), 3, (0, 0, 255), -1)
            cv2.putText(disp, f"frame:{frame_idx}  [q: quit | r: reselect ROI | e: save&exit]",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("tracking", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e'):          # ★ 즉시 저장 종료
                early_exit = True
                break
            if key == ord('r'):
                rx, ry, rw, rh = _select_roi_scaled(frame, "Re-select Drone", *display_max)
                if rw > 0 and rh > 0:
                    tracker = _create_kcf()
                    tracker.init(frame, (rx, ry, rw, rh))
                    x, y, w, h = rx, ry, rw, rh
                    box = (x, y, w, h)
                else:
                    if exit_on_cancel:
                        early_exit = True
                        break
                    # else: 취소했으면 그냥 기존 박스로 계속 진행

    cap.release()
    if visualize:
        try:
            cv2.destroyWindow("tracking")
        except Exception:
            pass

    # 저장 (early_exit 여부와 관계없이 save_csv가 지정돼 있으면 쓰기)
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

    parser = argparse.ArgumentParser(description="2-Stage Drone GPS Estimation (timestamp CSV)")
    parser.add_argument("--video", default="test_video.mp4", help="Path to the input video")
    parser.add_argument("--cones-gps", default="cones_gps.json", help="Path to the cones GPS JSON file")
    parser.add_argument("--model", default="best.pt", help="(unused) KCF model placeholder")
    parser.add_argument("--output-csv", default="drone_gps_output.csv", help="Path to the output CSV file")
    parser.add_argument("--start-time", default=None, help="Manual start time (e.g., \"2025-09-24 12:34:56.789\" or epoch seconds)")
    parser.add_argument("--exit-on-cancel", action="store_true",
                        help="If set, ESC-cancel in ROI window saves current results and exits")
    args = parser.parse_args()

    result = run_2stage_drag_kcf(
        video_path=args.video,
        cones_gps_json_path=args.cones_gps,
        start_time=args.start_time,
        save_csv=args.output_csv,
        visualize=True,
        exit_on_cancel=args.exit_on_cancel,
    )
    print("2-Stage processing completed.", result)
