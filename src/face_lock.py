import time
from collections import deque
from utils import iou, center_distance


class FaceLock:
    """
    Locks onto the same face across frames using bounding box similarity.
    """

    def __init__(self, iou_threshold=0.3, max_lost_frames=15):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.locked_box = None
        self.locked_since = None
        self.lost_counter = 0
        self.history = deque(maxlen=1000)

    def update(self, detected_boxes):
        """
        detected_boxes: list of (x, y, w, h)
        Returns: locked_box or None
        """
        now = time.time()

        # If nothing detected, possibly lose lock
        if not detected_boxes:
            if self.locked_box is not None:
                self.lost_counter += 1
                if self.lost_counter > self.max_lost_frames:
                    self._unlock()
            return self.locked_box

        # If no face locked yet → lock first one
        if self.locked_box is None:
            self._lock(detected_boxes[0])
            return self.locked_box

        # Try to match existing lock
        best_match = None
        best_score = 0

        for box in detected_boxes:
            score = iou(self.locked_box, box)
            if score > best_score:
                best_score = score
                best_match = box

        if best_score >= self.iou_threshold:
            self.locked_box = best_match
            self.lost_counter = 0
        else:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                self._unlock()

        return self.locked_box

    def _lock(self, box):
        self.locked_box = box
        self.locked_since = time.time()
        self.lost_counter = 0
        self.history.append(("LOCK", time.ctime(), box))

    def _unlock(self):
        if self.locked_box is not None:
            self.history.append(("UNLOCK", time.ctime(), self.locked_box))
        self.locked_box = None
        self.locked_since = None
        self.lost_counter = 0

    def get_lock_duration(self):
        if self.locked_since is None:
            return 0
        return time.time() - self.locked_since
