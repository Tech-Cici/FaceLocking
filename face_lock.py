import cv2
import time
import math
from pathlib import Path
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

class FaceLocker:
    def __init__(self, target_name: str, log_dir="history"):
        self.target_name = target_name
        self.locked = False
        self.locked_bbox = None
        self.prev_center = None
        self.lost_counter = 0
        self.LOCK_TIMEOUT = 30

        timestamp = time.strftime("%Y%m%d%H%M%S")
        Path(log_dir).mkdir(exist_ok=True)
        self.log_file = open(f"{log_dir}/{target_name}_history_{timestamp}.txt", "w")

    def log(self, action, desc):
        ts = time.strftime("%H:%M:%S")
        self.log_file.write(f"[{ts}] {action} - {desc}\n")
        self.log_file.flush()

    def lock(self, bbox):
        self.locked = True
        self.locked_bbox = bbox
        self.prev_center = None
        self.lost_counter = 0
        self.log("LOCK", "Face locked")

    def unlock(self):
        self.locked = False
        self.locked_bbox = None
        self.prev_center = None
        self.log("UNLOCK", "Face lost")

    def update_bbox(self, bbox):
        self.locked_bbox = bbox
        self.lost_counter = 0

    def mark_lost(self):
        self.lost_counter += 1
        if self.lost_counter > self.LOCK_TIMEOUT:
            self.unlock()

    def detect_actions(self, frame, face_landmarks, bbox):
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2

        if self.prev_center:
            dx = cx - self.prev_center[0]
            if dx > 15:
                self.log("MOVE_RIGHT", f"{dx}px")
            elif dx < -15:
                self.log("MOVE_LEFT", f"{dx}px")

        self.prev_center = (cx, cy)

        # --- Blink detection (EAR) ---
        lm = face_landmarks.landmark

        def dist(a, b):
            return math.hypot(a.x - b.x, a.y - b.y)

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        def eye_ratio(eye):
            return (dist(lm[eye[1]], lm[eye[5]]) + dist(lm[eye[2]], lm[eye[4]])) / (2.0 * dist(lm[eye[0]], lm[eye[3]]))

        ear = (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2
        if ear < 0.20:
            self.log("BLINK", f"EAR={ear:.2f}")

        # --- Smile detection ---
        mouth_width = dist(lm[61], lm[291])
        mouth_height = dist(lm[13], lm[14])
        ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        if ratio > 2.5:
            self.log("SMILE", f"ratio={ratio:.2f}")
