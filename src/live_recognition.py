import cv2
import sqlite3
import numpy as np
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from insightface.app import FaceAnalysis
from lib.database import DatabaseManager

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database_db/face_profiles.db"))

def load_known_faces(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Join profiles and face_encodings tables
    cursor.execute("""
        SELECT p.name, f.encoding
        FROM profiles p
        JOIN face_encodings f ON p.id = f.profile_id
    """)
    known_face_encodings = []
    known_face_names = []
    for name, encoding_blob in cursor.fetchall():
        try:
            encoding = pickle.loads(encoding_blob)
        except Exception as e:
            print(f"Error loading encoding for {name}: {e}")
            continue
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    conn.close()
    return known_face_encodings, known_face_names

def recognize_faces_from_stream(known_face_encodings, known_face_names, video_source=0):
    video_capture = cv2.VideoCapture(video_source)
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    db = DatabaseManager(DB_PATH)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        faces = app.get(frame)
        face_infos = []
        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            embedding = face.embedding
            # Compare to known embeddings using cosine similarity
            name = "Unknown"
            best_score = -1
            best_idx = -1
            for idx, known_emb in enumerate(known_face_encodings):
                if embedding.shape == known_emb.shape:
                    score = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            if best_score > 0.4 and best_idx != -1:
                name = known_face_names[best_idx]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Store info for possible saving
            # Estimate face angle (yaw) if available
            angle = 0
            if hasattr(face, 'pose') and face.pose is not None:
                # pose: [yaw, pitch, roll]
                angle = float(face.pose[0])
            elif hasattr(face, 'yaw'):  # Some versions may have .yaw
                angle = float(face.yaw)
            face_infos.append({
                'bbox': (left, top, right, bottom),
                'embedding': embedding,
                'face_img': frame[top:bottom, left:right],
                'angle': angle
            })
        cv2.imshow('Live Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            print(f"\n[INFO] Saving {len(face_infos)} detected faces...")
            for i, info in enumerate(face_infos):
                name = input(f"Enter name for Face {i+1} (or leave blank for 'Unknown'): ").strip() or "Unknown"
                profile_id = db.add_profile(name=name)
                db.add_face_encoding(profile_id, info['embedding'], angle=info['angle'])
                img_dir = os.path.join(os.path.dirname(DB_PATH), "added_faces")
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, f"face_{profile_id}.jpg")
                cv2.imwrite(img_path, info['face_img'])
                print(f"✓ Saved Face {i+1} as '{name}' (Profile ID: {profile_id}, Angle: {info['angle']:.1f})")
            print("[INFO] Done saving faces. Reloading known faces...")
            # Reload known faces so new ones are recognized immediately
            known_face_encodings[:], known_face_names[:] = load_known_faces(DB_PATH)
            print("[INFO] Reload complete. Press 'q' to quit or continue recognizing.")

if __name__ == "__main__":
    known_face_encodings, known_face_names = load_known_faces(DB_PATH)
    recognize_faces_from_stream(known_face_encodings, known_face_names)