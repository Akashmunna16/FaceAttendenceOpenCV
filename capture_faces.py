import cv2
import os

# ── Settings ──────────────────────────────────────────────
DATASET_DIR = "dataset"
SAMPLES_TO_CAPTURE = 54          # number of face images to save
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# ──────────────────────────────────────────────────────────

def capture_faces():
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Name cannot be empty!")
        return

    # Create a folder for this person
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    cam = cv2.VideoCapture(0)          # 0 = default webcam

    if not cam.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print(f"\n[INFO] Capturing {SAMPLES_TO_CAPTURE} images for '{name}'.")
    print("[INFO] Look at the camera. Press 'q' to quit early.\n")

    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            count += 1
            # Save the face region (grayscale)
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)

            # Draw rectangle & counter on screen
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing: {count}/{SAMPLES_TO_CAPTURE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Capture Faces  [press q to quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= SAMPLES_TO_CAPTURE:
            print(f"[INFO] Done! {count} images saved to '{person_dir}'")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()