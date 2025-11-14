import argparse
import queue
import threading
import time
import cv2
import grpc
import numpy as np
import drone_pb2, drone_pb2_grpc

class ControlQueue:
    def __init__(self, keepalive_interval=1.0):
        self.q = queue.Queue()
        self._closed = False
        self.keepalive_interval = keepalive_interval

    def push_key(self, k: str):
        self.q.put(drone_pb2.Control(key=drone_pb2.KeyEvent(key=k)))

    def push_roi(self, x, y, w, h):
        self.q.put(drone_pb2.Control(roi=drone_pb2.ROI(x=int(x), y=int(y), w=int(w), h=int(h))))

    def close(self):
        self._closed = True

    def gen(self):
        last = time.time()
        while not self._closed:
            try:
                msg = self.q.get(timeout=0.05)
                yield msg
            except queue.Empty:
                now = time.time()
                if now - last > self.keepalive_interval:
                    last = now
                    # keepalive ping (empty key)
                    yield drone_pb2.Control(key=drone_pb2.KeyEvent(key=""))

def main():
    need_auto_roi = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:50051")
    args = parser.parse_args()

    # Keepalive options (reduce “too_many_pings” errors)
    options = [
        ("grpc.keepalive_time_ms", 60000),
        ("grpc.keepalive_timeout_ms", 20000),
        ("grpc.keepalive_permit_without_calls", True),
        ("grpc.http2.max_pings_without_data", 5),
    ]
    channel = grpc.insecure_channel(args.server, options=options)
    stub = drone_pb2_grpc.DroneStreamStub(channel)

    cq = ControlQueue()
    win = "Drone Stream (r: ROI, q: exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    latest_frame = None
    latest_frame_lock = threading.Lock()
    running = True

    def select_roi_safe(snapshot, window_name="Select ROI"):
        # Auto-resize ROI window if too large (max 960px for long edge)
        h, w = snapshot.shape[:2]
        scale = min(960 / max(h, w), 1.0)
        if scale < 1.0:
            resized = cv2.resize(snapshot, (int(w * scale), int(h * scale)))
        else:
            resized = snapshot.copy()

        print(f"[Client] ROI window scale={scale:.3f}")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, resized.shape[1], resized.shape[0])

        roi_scaled = cv2.selectROI(
            window_name, resized, fromCenter=False, showCrosshair=True
        )
        cv2.destroyWindow(window_name)

        # Restore original coordinates
        x, y, rw, rh = map(int, roi_scaled)
        x, y, rw, rh = int(x / scale), int(y / scale), int(rw / scale), int(rh / scale)
        return (x, y, rw, rh)


    def recv_loop():
        nonlocal latest_frame
        nonlocal need_auto_roi

        try:
            for frame in stub.StreamVideo(cq.gen()):
                if frame.status:
                    print("[STATUS]", frame.status)

                    if frame.status == "tracking_lost":
                        # Trigger ROI mode automatically
                        need_auto_roi = True
                        cq.push_key('r')
                        continue

                if frame.jpg:
                    arr = np.frombuffer(frame.jpg, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    with latest_frame_lock:
                        latest_frame = img

        except grpc.RpcError as e:
            print("[gRPC]", e)



    t = threading.Thread(target=recv_loop, daemon=True)
    t.start()

    try:
        while running:
            with latest_frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is not None:
                cv2.imshow(win, frame)

            # Auto ROI mode when tracking lost
            if need_auto_roi:
                need_auto_roi = False
                print("[Client] Tracking lost — entering ROI mode automatically")

                # Switch server to ROI mode
                cq.push_key('r')

                # Close main window temporarily to avoid freeze
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                time.sleep(0.1)

                # Grab latest snapshot
                snap = None
                for _ in range(30):
                    with latest_frame_lock:
                        snap = None if latest_frame is None else latest_frame.copy()
                    if snap is not None:
                        break
                    time.sleep(0.01)

                if snap is None:
                    print("[Client] No frame available for ROI snapshot.")
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                else:
                    roi_win = "Select ROI (Enter=apply, Esc=cancel)"
                    roi = cv2.selectROI(roi_win, snap, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow(roi_win)

                    x, y, w, h = map(int, roi)
                    if w > 0 and h > 0:
                        cq.push_roi(x, y, w, h)
                        print(f"[Client] ROI applied: {x},{y},{w},{h}")
                        time.sleep(0.05)
                    else:
                        print("[Client] ROI canceled.")

                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

                continue

            # Manual key handling
            k = cv2.waitKey(10) & 0xFF
            if k == ord('r'):
                cq.push_key('r')

                snap = None
                for _ in range(30):
                    with latest_frame_lock:
                        snap = None if latest_frame is None else latest_frame.copy()
                    if snap is not None:
                        break
                    time.sleep(0.01)

                if snap is None:
                    print("[Client] No frame available for ROI snapshot.")
                    continue

                roi_win = "Select ROI (Enter=apply, Esc=cancel)"
                cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)
                cv2.imshow(roi_win, snap)
                roi = cv2.selectROI(roi_win, snap, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(roi_win)

                x, y, w, h = map(int, roi)
                if w > 0 and h > 0:
                    cq.push_roi(x, y, w, h)
                    print(f"[Client] ROI applied: {x},{y},{w},{h}")
                    time.sleep(0.05)
                else:
                    print("[Client] ROI canceled.")

            elif k == ord('q'):
                print("[Client] Exiting")
                cq.push_key('q')
                time.sleep(0.1)
                cq.close()
                break

    finally:
        cv2.destroyAllWindows()
        t.join(timeout=1.0)

if __name__ == "__main__":
    main()
