import cv2
import os
import sys
import time
import numpy as np
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from insightface.app import FaceAnalysis
from lib.database import DatabaseManager
from lib.face_analyzer import FaceAnalyzer
import platform

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database_db/face_profiles.db"))


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print("[INFO] PyTorch CUDA available. Using GPU.")
            return 0
    except Exception:
        pass
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("[INFO] ONNXRuntime CUDA available. Using GPU.")
            return 0
    except Exception:
        pass
    print("[WARNING] No GPU detected. Using CPU (ctx_id=-1). For best performance, install CUDA drivers and GPU-enabled PyTorch/ONNXRuntime.")
    return -1


def print_onnxruntime_providers():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNXRuntime available providers: {providers}")
        if 'CUDAExecutionProvider' not in providers:
            print("[INFO] ONNXRuntime is not using CUDAExecutionProvider. Models will run on CPU.")
            print("To enable GPU support with ONNXRuntime, install onnxruntime-gpu matching your CUDA version.")
            print("Example (pip): pip install onnxruntime-gpu")
            print("Example (conda, recommended for CUDA toolkits): conda install -c conda-forge onnxruntime-gpu")
    except Exception:
        print("[INFO] onnxruntime is not installed in this environment. If you rely on onnx models, consider installing onnxruntime or onnxruntime-gpu.")


def select_video_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        HAS_TKINTER = True
    except ImportError:
        HAS_TKINTER = False
    if HAS_TKINTER:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        root.destroy()
        return file_path if file_path else None
    else:
        return input("Enter path to video file: ").strip()


def batch_cosine_similarity(embedding, embeddings):
    embedding = embedding.astype(np.float32)
    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    dot = np.dot(embeddings, embedding)
    norm_emb = np.linalg.norm(embeddings, axis=1)
    norm_query = np.linalg.norm(embedding)
    return dot / (norm_emb * norm_query + 1e-8)


def get_pose(face):
    if hasattr(face, 'pose') and face.pose is not None:
        return tuple(face.pose)
    return (0.0, 0.0, 0.0)


def is_duplicate_unknown(embedding, all_unknown_embeddings, threshold=0.2):
    if len(all_unknown_embeddings) == 0:
        return False
    scores = batch_cosine_similarity(embedding, all_unknown_embeddings)
    return np.any(scores > threshold)

def reload_profiles(profiles=[], known_face_encodings=[], known_face_ids=[], db=None):
    # nonlocal profiles, known_face_encodings, known_face_ids
    profiles.clear()
    known_face_encodings.clear()
    known_face_ids.clear()
    profiles.extend(db.get_all_profiles())
    for profile in profiles:
        encodings = db.get_profile_encodings(profile['id'])
        for emb, angle in encodings:
            known_face_encodings.append(emb.astype(np.float32))
            known_face_ids.append(profile['id'])
            
def main():
    parser = argparse.ArgumentParser(description="Face Auto-Enroll Video Recognition")
    parser.add_argument('--video', type=str, default=None, help='Path to video file (if not provided, dialog will open)')
    parser.add_argument('--db', type=str, default="database_db/footage_faces.db", help='Path to database file')
    parser.add_argument('--thresh', type=float, default=0.2, help='Matching threshold (default: 0.2)')
    parser.add_argument('--window', type=str, default='Video Auto-Enroll', help='Window name')
    parser.add_argument('--demographics', action='store_true', help='Enable InsightFace/FairFace demographics for new faces')
    parser.add_argument('--menu', action='store_true', help='Show help menu overlay')
    args = parser.parse_args()

    print("\n[Face Auto-Enroll] Keyboard controls:")
    print("  q: Quit")
    print("  p: Pause/Resume")
    print("  s: Skip frame")
    print("  h: Show help")
    print("  u: Update cached database (reload profiles)")

    # Video selection
    video_path = args.video
    if not video_path:
        video_path = select_video_file()
    if not video_path or not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)

    # Set model cache directories for InsightFace assets
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.environ['INSIGHTFACE_HOME'] = models_dir
    os.environ['ONNX_HOME'] = models_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(models_dir, "huggingface")

    db_path = args.db
    db = DatabaseManager(db_path=db_path)
    print(f"Using database at: {db_path}")
    ctx_id = check_gpu()
    print_onnxruntime_providers()

    app = FaceAnalysis()
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    analyzer = FaceAnalyzer()

    profiles = []
    known_face_encodings = []
    known_face_ids = []
    reload_profiles(profiles, known_face_encodings=known_face_encodings, known_face_ids=known_face_ids, db=db)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    unknown_faces = []
    all_unknown_embeddings = []
    paused = False

    prev_time = time.time()
    fps = 0.0

    window_name = args.window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = None
        if not paused:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_count += 1
            faces = app.get(frame)
            for face in faces:
                embedding = face.embedding
                bbox = face.bbox.astype(int)
                left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
                pose = get_pose(face)

                h, w = frame.shape[:2]
                top_c = max(0, min(top, h - 1))
                bottom_c = max(0, min(bottom, h))
                left_c = max(0, min(left, w - 1))
                right_c = max(0, min(right, w))
                cropped = frame[top_c:bottom_c, left_c:right_c]

                match_found = False
                matched_profile_id = None

                if len(known_face_encodings) > 0:
                    scores = batch_cosine_similarity(embedding, np.array(known_face_encodings))
                    best_score = float(np.max(scores))
                    best_idx = int(np.argmax(scores))
                    if best_score > args.thresh:
                        match_found = True
                        matched_profile_id = known_face_ids[best_idx]
                        existing_encodings = db.get_profile_encodings(matched_profile_id)
                        pose_threshold = 30.0
                        is_new_angle = True
                        for _, existing_angle in existing_encodings:
                            try:
                                if abs(pose[0] - float(existing_angle)) < pose_threshold:
                                    is_new_angle = False
                                    break
                            except Exception:
                                pass
                        if is_new_angle and (abs(pose[0]) > pose_threshold or abs(pose[1]) > pose_threshold):
                            db.add_face_encoding(matched_profile_id, embedding, angle=float(pose[0]))
                            known_face_encodings.append(embedding.astype(np.float32))
                            known_face_ids.append(matched_profile_id)
                            print(f"[Frame {frame_count}] Added new angle ({pose}) embedding to profile {matched_profile_id}")

                # Only compute demographics for new faces using fast InsightFace attributes
                if not match_found and not is_duplicate_unknown(embedding, all_unknown_embeddings, threshold=0.2):
                    # Use age/gender from InsightFace (fast, GPU-accelerated)
                    age = str(getattr(face, 'age', '-'))
                    gender = str(getattr(face, 'gender', '-'))
                    ethnicity = '-'  # Ethnicity can be computed later in batch
                else:
                    # For matched faces, skip demographics (already in DB)
                    age = gender = ethnicity = '-'

                match_found = False
                matched_profile_id = None

                if len(known_face_encodings) > 0:
                    scores = batch_cosine_similarity(embedding, np.array(known_face_encodings))
                    best_score = float(np.max(scores))
                    best_idx = int(np.argmax(scores))
                    if best_score > args.thresh:
                        match_found = True
                        matched_profile_id = known_face_ids[best_idx]
                        existing_encodings = db.get_profile_encodings(matched_profile_id)
                        pose_threshold = 30.0
                        is_new_angle = True
                        for _, existing_angle in existing_encodings:
                            try:
                                if abs(pose[0] - float(existing_angle)) < pose_threshold:
                                    is_new_angle = False
                                    break
                            except Exception:
                                pass
                        if is_new_angle and (abs(pose[0]) > pose_threshold or abs(pose[1]) > pose_threshold):
                            db.add_face_encoding(matched_profile_id, embedding, angle=float(pose[0]))
                            known_face_encodings.append(embedding.astype(np.float32))
                            known_face_ids.append(matched_profile_id)
                            print(f"[Frame {frame_count}] Added new angle ({pose}) embedding to profile {matched_profile_id}")

                if not match_found and not is_duplicate_unknown(embedding, all_unknown_embeddings, threshold=0.2):
                    profile_id = db.add_profile(
                        name=f"Unknown_{frame_count}",
                        gender=str(gender),
                        age_range=str(age),
                        ethnicity=str(ethnicity),
                        notes=f"Auto-enrolled from video frame {frame_count}"
                    )
                    db.add_face_encoding(profile_id, embedding, angle=float(pose[0]))
                    img_dir = os.path.join(os.path.dirname(db_path), "added_faces")
                    os.makedirs(img_dir, exist_ok=True)
                    img_path = os.path.join(img_dir, f"face_{profile_id}.jpg")
                    if cropped.size > 0 and (bottom_c > top_c) and (right_c > left_c):
                        cv2.imwrite(img_path, cropped)
                        unknown_faces.append((profile_id, img_path))
                    else:
                        print(f"[Frame {frame_count}] Warning: Skipped saving empty/invalid face crop for profile {profile_id} (bbox: {left},{top},{right},{bottom})")
                    all_unknown_embeddings.append(embedding)
                    print(f"[Frame {frame_count}] Added unknown face as profile {profile_id}")

                label = f"Frame:{frame_count} "
                if match_found:
                    label += f"ID:{matched_profile_id}"
                    # Get demographics from DB profile
                    profile = next((p for p in profiles if p['id'] == matched_profile_id), None)
                    db_gender = profile['gender'] if profile and profile.get('gender') else '-'
                    db_age = profile['age_range'] if profile and profile.get('age_range') else '-'
                    db_ethnicity = profile['ethnicity'] if profile and profile.get('ethnicity') else '-'
                    label += f" | G:{db_gender} A:{db_age} E:{db_ethnicity}"
                else:
                    label += f"ID:New"
                    label += f" | G:{gender} A:{age} E:{ethnicity}"
                label += f" | Pose:{pose[0]:.1f}"

                edge_margin = 0.15
                near_edge = (left < w * edge_margin or right > w * (1 - edge_margin))
                low_conf = near_edge and abs(pose[0]) > 60
                if low_conf:
                    label += " | LowConf"

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if match_found else (0, 0, 255), 2)
                # Center the label above the bounding box
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                center_x = left + (right - left) // 2
                text_x = max(0, center_x - label_width // 2)
                text_y = max(0, top - 10)
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS
        now = time.time()
        dt = now - prev_time if prev_time else 0.001
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        prev_time = now

        help_text = f"[q] Quit  [p] Pause/Resume  [s] Skip frame  [h] Help  | FPS: {fps:.1f}"
        if frame is not None:
            if args.menu:
                cv2.putText(frame, help_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            continue
        elif key == ord('h'):
            print("\n[HELP] Keyboard controls:")
            print("  q: Quit")
            print("  p: Pause/Resume")
            print("  s: Skip frame")
            print("  h: Show help")
            print("  u: Update cached database (reload profiles)")
        elif key == ord('u'):
            print("[INFO] Reloading profiles and encodings from database...")
            reload_profiles(profiles, known_face_encodings=known_face_encodings, known_face_ids=known_face_ids, db=db)

    video_capture.release()
    cv2.destroyAllWindows()

    print("\nAuto-enroll complete. Unknown faces added:")
    for pid, img_path in unknown_faces:
        print(f"Profile ID: {pid}, Image: {img_path}")
    print("You can now relabel these faces in the database or via a review tool.")


## Removed duplicate/broken select_video_file

if __name__ == "__main__":
    main()