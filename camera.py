import cv2
import mediapipe as mp
from face_lock import FaceLocker
from utils import iou

TARGET_NAME = "gabi"   # manually chosen identity

locker = FaceLocker(TARGET_NAME)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    matched = False

    for (x, y, w, h) in faces:
        bbox = (x, y, w, h)

        name = TARGET_NAME   # ← simulated recognition (replace later with ArcFace)

        if not locker.locked:
            if name == TARGET_NAME:
                locker.lock(bbox)

        if locker.locked:
            if iou(bbox, locker.locked_bbox) > 0.3:
                locker.update_bbox(bbox)
                matched = True
                color = (0, 255, 0)
                label = f"{name} [LOCKED]"
            else:
                continue
        else:
            color = (255, 0, 0)
            label = name

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if locker.locked:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                locker.detect_movement(bbox)
                locker.detect_blink_and_smile(results.multi_face_landmarks[0])

    if locker.locked and not matched:
        locker.mark_lost()

    cv2.imshow("Face Locking System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
