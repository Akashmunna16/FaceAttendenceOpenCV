import cv2
import numpy as np
import os
import pickle
import csv
from datetime import datetime

# ── Settings ──────────────────────────────────────────────
TRAINER_FILE    = os.path.join("trainer", "trainer.yml")
LABELS_FILE     = os.path.join("trainer", "labels.pkl")
ATTENDANCE_FILE = "attendance.csv"
HAAR_CASCADE    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 70   # lower = stricter; raise if too many unknowns
# ──────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
        print("ERROR: Trained model not found. Run Step 2 first.")
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    with open(LABELS_FILE, "rb") as f:
        id_to_name = pickle.load(f)

    return recognizer, id_to_name


def mark_attendance(name, attendance_log):
    """Write a row to CSV only if the person hasn't been marked this session."""
    if name in attendance_log:
        return   # already marked

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Append to CSV (create with header if new)
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, date_str, time_str])

    attendance_log.add(name)
    print(f"[ATTENDANCE] ✅  {name}  marked at {time_str}")


def recognize():
    recognizer, id_to_name = load_model()
    if recognizer is None:
        return

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("\n[INFO] Starting real-time recognition …")
    print("[INFO] Press 'q' to quit.\n")

    attendance_log = set()   # tracks who has been marked this session

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            label_id, confidence = recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name  = id_to_name.get(label_id, "Unknown")
                color = (0, 255, 0)          # green  → recognised
                text  = f"{name}  ({confidence:.1f})"
                mark_attendance(name, attendance_log)
            else:
                name  = "Unknown"
                color = (0, 0, 255)          # red    → unknown
                text  = "Unknown"

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, text, (x+5, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show how many people are marked today
        info = f"Marked today: {len(attendance_log)}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)

        cv2.imshow("Face Attendance  [press q to quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session ended. Attendance saved to '{ATTENDANCE_FILE}'")


if __name__ == "__main__":
    recognize()