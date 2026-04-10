import cv2
import numpy as np
import os
import pickle

# ── Settings ──────────────────────────────────────────────
DATASET_DIR  = "dataset"
TRAINER_DIR  = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE  = os.path.join(TRAINER_DIR, "labels.pkl")   # id <-> name map
# ──────────────────────────────────────────────────────────

def get_images_and_labels(dataset_dir):
    """Walk dataset/ and return (images, labels, id_to_name map)."""
    face_samples = []
    labels       = []
    id_to_name   = {}
    current_id   = 0

    for person_name in sorted(os.listdir(dataset_dir)):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        id_to_name[current_id] = person_name
        print(f"  → ID {current_id}  :  {person_name}")

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            try:
                # Load as grayscale PIL image, convert to numpy
                pil_img  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_array = np.array(pil_img, dtype="uint8")
                face_samples.append(img_array)
                labels.append(current_id)
            except Exception as e:
                print(f"    [WARN] Skipping {img_file}: {e}")

        current_id += 1

    return face_samples, labels, id_to_name


def train():
    print("\n[INFO] Reading training data …")

    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"ERROR: No data found in '{DATASET_DIR}/'. Run Step 1 first.")
        return

    faces, labels, id_to_name = get_images_and_labels(DATASET_DIR)

    print(f"\n[INFO] Training on {len(faces)} images for {len(id_to_name)} person(s) …")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    os.makedirs(TRAINER_DIR, exist_ok=True)
    recognizer.write(TRAINER_FILE)

    # Save the id→name dictionary so Step 3 can look up names
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(id_to_name, f)

    print(f"[INFO] Model saved  →  {TRAINER_FILE}")
    print(f"[INFO] Labels saved →  {LABELS_FILE}")
    print("\n✅  Training complete! Now run Step 3.")

if __name__ == "__main__":
    train()