import os
import math
import time
import logging
import warnings
from collections import deque
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Suppress external library and YOLO logging
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Automatically mount Google Drive when running inside Colab.
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "/content/drive/MyDrive/Amrendra_Guardex/throwing_counter/merged_final_dataset/runs/throwing_box_finetune_v1/weights/best.pt"
INPUT_VIDEO_PATH = "/content/drive/MyDrive/Amrendra_Guardex/throwing_counter/input_videos/day.MP4"
OUTPUT_DIR = "/content/drive/MyDrive/Amrendra_Guardex/throwing_counter/throwing_dino_finetunned"

# Line Placement for Dual-Line Counting
LINE_1_COORDS = [(830, 129), (850, 1423)]   # Primary line
LINE_2_COORDS = [(500, 129), (520, 1423)]   # Backup line

TARGET_FPS = 10
START_TIME = None
END_TIME = None

CONF = 0.35              # Base detector confidence
NMS_IOU = 0.5
IMGSZ = None
USE_ROI_MASK = True

# Permissive Counting Logic
MIN_FRAMES_TO_COUNT = 2
MIN_LEFTWARD_DIST_PX = 50
MIN_PEAK_STEP_VX_PX = 0
MIN_PEAK_CONF_TO_COUNT = 0.0

# Momentum Tracking Parameters
ALPHA = 0.6
MAX_JUMP_FRAC = 0.50
BACK_TOL_FRAC = 0.03
Y_GATE_FRAC = 0.14
X_SCALE_FRAC = 0.16
MAX_SIZE_RATIO = 3.2
MAX_MISS_SEC = 0.4

SAVE_COUNT_LOG_FILE = True


def parse_time_to_seconds(time_val):
    if time_val is None:
        return None
    if isinstance(time_val, (int, float)):
        return float(time_val)
    if isinstance(time_val, str):
        parts = [float(p) for p in time_val.strip().split(":")]
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        raise ValueError(f"Invalid time format string: {time_val}")
    raise TypeError(f"Unsupported time format: {type(time_val)}")


class MomentumTrack:
    def __init__(self, track_id, det, frame_idx):
        self.id = track_id
        self.cx, self.cy, self.x1, self.y1, self.x2, self.y2, self.conf = det
        self.start_cx = self.cx
        self.start_cy = self.cy
        self.max_conf = float(self.conf)
        self.max_step_dx = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.has_vel = False
        self.last_f = frame_idx
        self.hits = 1
        self.history = deque([(self.cx, self.cy)], maxlen=20)
        self.counted = False
        self.crossed_l1 = False
        self.crossed_l2 = False

    @property
    def w(self):
        return max(1, self.x2 - self.x1)

    @property
    def h(self):
        return max(1, self.y2 - self.y1)

    def predict(self, frame_idx):
        dt = frame_idx - self.last_f
        return self.cx + self.vx * dt, self.cy + self.vy * dt

    def update(self, det, frame_idx, alpha):
        prev = (self.cx, self.cy, self.x1)
        cx, cy, x1, y1, x2, y2, conf = det
        dt = max(1, frame_idx - self.last_f)

        step_leftward = self.cx - cx
        if step_leftward > self.max_step_dx:
            self.max_step_dx = float(step_leftward)

        meas_vx = (cx - self.cx) / dt
        meas_vy = (cy - self.cy) / dt
        if self.has_vel:
            self.vx = alpha * meas_vx + (1 - alpha) * self.vx
            self.vy = alpha * meas_vy + (1 - alpha) * self.vy
        else:
            self.vx, self.vy = meas_vx, meas_vy
            self.has_vel = True

        self.cx, self.cy, self.x1, self.y1, self.x2, self.y2, self.conf = det
        if float(conf) > self.max_conf:
            self.max_conf = float(conf)
        self.last_f = frame_idx
        self.hits += 1
        self.history.append((cx, cy))
        return prev


class MomentumDualLineBoxCounter:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.use_half = torch.cuda.is_available()
        self.device = 0 if self.use_half else "cpu"

        self.cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video at {INPUT_VIDEO_PATH}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.orig_fps <= 0 or np.isnan(self.orig_fps):
            self.orig_fps = 25.0
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_sec = parse_time_to_seconds(START_TIME)
        end_sec = parse_time_to_seconds(END_TIME)
        self.start_frame = int(start_sec * self.orig_fps) if start_sec is not None else 0
        self.start_frame = max(0, min(self.start_frame, self.total_video_frames - 1))
        self.end_frame = int(end_sec * self.orig_fps) if end_sec is not None else self.total_video_frames
        self.end_frame = max(self.start_frame + 1, min(self.end_frame, self.total_video_frames))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.fps = min(float(TARGET_FPS), self.orig_fps) if TARGET_FPS else self.orig_fps
        self.step = self.orig_fps / self.fps

        self.log_file = None
        if SAVE_COUNT_LOG_FILE:
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                video_stem = os.path.splitext(os.path.basename(INPUT_VIDEO_PATH))[0]
                count_log_path = os.path.join(OUTPUT_DIR, f"{video_stem}_count_log_ist.txt")
                self.log_file = open(count_log_path, "w", encoding="utf-8")
            except Exception:
                self.log_file = None

        self.l1_p1, self.l1_p2 = self._normalize_line_points(LINE_1_COORDS)
        self.l2_p1, self.l2_p2 = self._normalize_line_points(LINE_2_COORDS)
        if (self.l1_p1[0] + self.l1_p2[0]) < (self.l2_p1[0] + self.l2_p2[0]):
            self.l1_p1, self.l2_p1 = self.l2_p1, self.l1_p1
            self.l1_p2, self.l2_p2 = self.l2_p2, self.l1_p2

        self.ROI_LEFT_X = int(self.width * 0.05)
        self.ROI_RIGHT_X = int(self.width * 0.62)
        self.MIN_BOX_W = int(self.width * 0.025)
        self.MIN_BOX_H = int(self.height * 0.025)

        self.MAX_JUMP = self.width * MAX_JUMP_FRAC
        self.BACK_TOL = self.width * BACK_TOL_FRAC
        self.Y_GATE = self.height * Y_GATE_FRAC
        self.X_SCALE = self.width * X_SCALE_FRAC
        self.MAX_MISS = max(2, int(round(MAX_MISS_SEC * self.fps)))

        self.tracks = {}
        self.next_id = 1
        self.counted_ids = set()
        self.total_count = 0
        self.frame_idx = 0
        self.raw_idx = self.start_frame
        self.cur_raw_idx = self.start_frame

    @staticmethod
    def _normalize_line_points(coords):
        p1, p2 = coords
        if p1[1] <= p2[1]:
            return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
        return (int(p2[0]), int(p2[1])), (int(p1[0]), int(p1[1]))

    @staticmethod
    def _side(pt, lp1, lp2):
        x, y = pt
        x1, y1 = lp1
        x2, y2 = lp2
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def _side1(self, pt):
        return self._side(pt, self.l1_p1, self.l1_p2)

    def _side2(self, pt):
        return self._side(pt, self.l2_p1, self.l2_p2)

    def _detect(self, frame):
        inp = frame.copy()
        if USE_ROI_MASK:
            inp[:, :self.ROI_LEFT_X] = 0
            inp[:, self.ROI_RIGHT_X:] = 0
        kwargs = dict(conf=CONF, iou=NMS_IOU, half=self.use_half, device=self.device, verbose=False)
        if IMGSZ:
            kwargs["imgsz"] = IMGSZ
        r = self.model.predict(inp, **kwargs)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if (x2 - x1) < self.MIN_BOX_W or (y2 - y1) < self.MIN_BOX_H:
                    continue
                dets.append(((x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2, float(c)))
        return dets

    def _match_cost(self, trk, det):
        cx, cy, x1, y1, x2, y2, _ = det
        dt = max(1, self.frame_idx - trk.last_f)

        fwd = trk.cx - cx
        if fwd < -self.BACK_TOL * dt or fwd > self.MAX_JUMP * dt:
            return None

        w, h = max(1, x2 - x1), max(1, y2 - y1)
        size_term = abs(math.log(w / trk.w)) + abs(math.log(h / trk.h))
        if size_term > 2 * math.log(MAX_SIZE_RATIO):
            return None

        if trk.has_vel:
            px, py = trk.predict(self.frame_idx)
            ey = cy - py
            if abs(ey) > self.Y_GATE + abs(trk.vy) * 0.5:
                return None
            ex = cx - px
            return (ex / self.X_SCALE) ** 2 + (ey / self.Y_GATE) ** 2 + 0.3 * size_term

        ey = cy - trk.cy
        if abs(ey) > self.Y_GATE:
            return None
        return (ey / self.Y_GATE) ** 2 + 0.2 * (abs(fwd) / (self.MAX_JUMP * dt)) + 0.3 * size_term

    def _associate(self, tracks, dets):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        BIG = 1e6
        cost = np.full((len(tracks), len(dets)), BIG, dtype=np.float64)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                c = self._match_cost(t, d)
                if c is not None:
                    cost[i, j] = c
        rows, cols = linear_sum_assignment(cost)
        matches, used_t, used_d = [], set(), set()
        for i, j in zip(rows, cols):
            if cost[i, j] < BIG:
                matches.append((i, j))
                used_t.add(i)
                used_d.add(j)
        return (matches,
                [i for i in range(len(tracks)) if i not in used_t],
                [j for j in range(len(dets)) if j not in used_d])

    def _register(self, trk, reason):
        if trk.counted or trk.id in self.counted_ids:
            return
        trk.counted = True
        self.counted_ids.add(trk.id)
        self.total_count += 1

        # Keep log in IST only when count increases
        now_ist = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
        log_line = f"[{now_ist}] Count: {self.total_count}"
        print(log_line, flush=True)
        if self.log_file is not None:
            self.log_file.write(log_line + '\n')
            self.log_file.flush()

    def _check_crossing(self, trk, prev):
        if trk.counted:
            return

        pcx, pcy, _ = prev

        if trk.cx < pcx:
            if (self._side1((pcx, pcy)) < 0 and self._side1((trk.cx, trk.cy)) >= 0) or (
                self._side1((trk.start_cx, trk.start_cy)) < 0 and self._side1((trk.cx, trk.cy)) >= 0
            ):
                trk.crossed_l1 = True

            if (self._side2((pcx, pcy)) < 0 and self._side2((trk.cx, trk.cy)) >= 0) or (
                self._side2((trk.start_cx, trk.start_cy)) < 0 and self._side2((trk.cx, trk.cy)) >= 0
            ):
                trk.crossed_l2 = True

        net_leftward_travel = trk.start_cx - trk.cx
        if (trk.hits < MIN_FRAMES_TO_COUNT
            or net_leftward_travel < MIN_LEFTWARD_DIST_PX
            or trk.max_step_dx < MIN_PEAK_STEP_VX_PX
            or trk.max_conf < MIN_PEAK_CONF_TO_COUNT):
            return

        if trk.crossed_l1:
            self._register(trk, "L1")
            return

        if trk.crossed_l2:
            self._register(trk, "L2")
            return

    def run(self):
        next_target = float(self.start_frame)
        try:
            while True:
                if self.raw_idx >= self.end_frame:
                    break
                if self.raw_idx < int(round(next_target)):
                    if not self.cap.grab():
                        break
                    self.raw_idx += 1
                    continue

                ok, frame = self.cap.read()
                if not ok:
                    break
                self.cur_raw_idx = self.raw_idx
                self.raw_idx += 1
                next_target += self.step
                self.frame_idx += 1

                dets = self._detect(frame)
                tracks = list(self.tracks.values())
                matches, lost_t, new_d = self._associate(tracks, dets)

                for ti, di in matches:
                    trk = tracks[ti]
                    prev = trk.update(dets[di], self.frame_idx, ALPHA)
                    self._check_crossing(trk, prev)

                for di in new_d:
                    trk = MomentumTrack(self.next_id, dets[di], self.frame_idx)
                    self.tracks[trk.id] = trk
                    self.next_id += 1

                for tid in [t.id for t in self.tracks.values() if self.frame_idx - t.last_f > self.MAX_MISS]:
                    del self.tracks[tid]
        finally:
            self.cap.release()
            if self.log_file is not None:
                self.log_file.close()


if __name__ == "__main__":
    MomentumDualLineBoxCounter().run()
