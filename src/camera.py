import cv2
import mediapipe as mp
from face_lock import FaceLock



def main():
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    locker = FaceLock()

    print("🎥 Face Locking System Started")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        boxes = []
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((x, y, bw, bh))

        locked_box = locker.update(boxes)

        # Draw detections
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

        # Draw locked face
        if locked_box:
            x, y, bw, bh = locked_box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            duration = locker.get_lock_duration()
            cv2.putText(frame, f"LOCKED {duration:.1f}s", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Locking", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
