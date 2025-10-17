import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle
from PIL import Image
import io
import base64
from collections import defaultdict
import time
import argparse
import shutil

# Face recognition libraries
from insightface.app import FaceAnalysis

# Database management
import sqlite3
from pathlib import Path

# For video processing
import tempfile

from lib.database import DatabaseManager
from lib.face_detector import FaceDetector
from lib.face_analyzer import FaceAnalyzer

# --- Matching utilities (aligned with video_auto_enroll.py) ---
def batch_cosine_similarity(embedding, embeddings):
    embedding = embedding.astype(np.float32)
    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    dot = np.dot(embeddings, embedding)
    norm_emb = np.linalg.norm(embeddings, axis=1)
    norm_query = np.linalg.norm(embedding)
    return dot / (norm_emb * norm_query + 1e-8)


def is_duplicate_unknown(embedding, all_unknown_embeddings, threshold=0.2):
    if len(all_unknown_embeddings) == 0:
        return False
    scores = batch_cosine_similarity(embedding, np.array(all_unknown_embeddings))
    return bool(np.any(scores > threshold))


def ensure_db_cache(db):
    """Ensure profiles and encodings are cached in session_state."""
    if 'profiles_cache' not in st.session_state:
        st.session_state['profiles_cache'] = db.get_all_profiles()
    if 'known_face_encodings' not in st.session_state:
        st.session_state['known_face_encodings'] = []
    if 'known_face_ids' not in st.session_state:
        st.session_state['known_face_ids'] = []
    if not st.session_state['known_face_encodings'] or not st.session_state['known_face_ids']:
        st.session_state['known_face_encodings'].clear()
        st.session_state['known_face_ids'].clear()
        for profile in st.session_state['profiles_cache']:
            encodings = db.get_profile_encodings(profile['id'])
            for emb, angle in encodings:
                st.session_state['known_face_encodings'].append(emb.astype(np.float32))
                st.session_state['known_face_ids'].append(profile['id'])

# Initialize face analysis model
@st.cache_resource
def load_face_analysis_model():
    """Load the face analysis model"""
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Face recognition class
class FaceRecognitionSystem:
    def __init__(self):
        self.app = load_face_analysis_model()
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.face_data = defaultdict(list)
        self.attribute_analyzer = FaceAnalyzer()
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from database_db"""
        conn = sqlite3.connect('face_profiles.db')
        c = conn.cursor()

        # Get all profiles and their encodings
        c.execute('''
        SELECT p.id, p.name, fe.encoding, fe.angle
        FROM profiles p
        JOIN face_encodings fe ON p.id = fe.profile_id
        ORDER BY p.id
        ''')

        results = c.fetchall()
        conn.close()

        if results:
            current_id = None
            for profile_id, name, encoding_blob, angle in results:
                if current_id != profile_id:
                    self.known_face_ids.append(profile_id)
                    self.known_face_names.append(name)
                    self.known_face_encodings.append([])
                    current_id = profile_id

                encoding = pickle.loads(encoding_blob)
                self.known_face_encodings[-1].append((encoding, angle))

    def detect_faces(self, image):
        """Detect faces in an image"""
        faces = self.app.get(image)
        return faces

    def analyze_face_attributes(self, face_img):
        """Analyze face attributes like age, gender, ethnicity"""
        try:
            return self.attribute_analyzer.analyze_face(face_img)
        except Exception as e:
            print(f"Error analyzing face attributes: {e}")
            return {
                'age': 'Unknown',
                'gender': 'Unknown',
                'ethnicity': 'Unknown'
            }

    def get_face_encoding(self, face_img, face_info):
        """Extract face encoding using InsightFace embedding."""
        try:
            embedding = getattr(face_info, 'embedding', None)
            if embedding is None:
                return None
            return np.asarray(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None

    def match_face(self, face_encoding, threshold=0.6):
        """Match a face encoding against known faces"""
        if not self.known_face_encodings:
            return None, 0

        best_match_id = None
        best_match_name = "Unknown"
        best_distance = float('inf')

        for i, encodings in enumerate(self.known_face_encodings):
            for known_encoding, _ in encodings:
                # Calculate distance
                if isinstance(face_encoding, np.ndarray) and isinstance(known_encoding, np.ndarray):
                    if face_encoding.shape != known_encoding.shape:
                        continue

                    distance = np.linalg.norm(face_encoding - known_encoding)

                    if distance < best_distance and distance < threshold:
                        best_distance = distance
                        best_match_id = self.known_face_ids[i]
                        best_match_name = self.known_face_names[i]

        return best_match_id, best_match_name

    def add_face_to_database(self, face_img, face_info, name, angle=0):
        """Add a new face to the database_db"""
        # Get face encoding
        face_encoding = self.get_face_encoding(face_img, face_info)
        if face_encoding is None:
            return None

        # Analyze face attributes
        attributes = self.analyze_face_attributes(face_img)

        # Add to database_db
        conn = sqlite3.connect('face_profiles.db')
        c = conn.cursor()

        # Insert new profile
        c.execute('''
        INSERT INTO profiles (name, created_at, last_seen, gender, age_range, ethnicity, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            name,
            datetime.now(),
            datetime.now(),
            attributes['gender'],
            str(attributes['age']),
            attributes['ethnicity'],
            "Initial registration"
        ))

        profile_id = c.lastrowid

        # Insert face encoding
        encoding_blob = pickle.dumps(face_encoding)
        c.execute('''
        INSERT INTO face_encodings (profile_id, encoding, angle)
        VALUES (?, ?, ?)
        ''', (profile_id, encoding_blob, angle))

        conn.commit()
        conn.close()

        # Update in-memory data
        self.known_face_ids.append(profile_id)
        self.known_face_names.append(name)
        self.known_face_encodings.append([(face_encoding, angle)])

        return profile_id

    def update_face_database(self, profile_id, face_img, face_info, angle):
        """Update a face in the database_db with a new angle"""
        # Get face encoding
        face_encoding = self.get_face_encoding(face_img, face_info)
        if face_encoding is None:
            return False

        # Add to database_db
        conn = sqlite3.connect('face_profiles.db')
        c = conn.cursor()

        # Insert new face encoding
        encoding_blob = pickle.dumps(face_encoding)
        c.execute('''
        INSERT INTO face_encodings (profile_id, encoding, angle)
        VALUES (?, ?, ?)
        ''', (profile_id, encoding_blob, angle))

        # Update last seen
        c.execute('''
        UPDATE profiles
        SET last_seen = ?
        WHERE id = ?
        ''', (datetime.now(), profile_id))

        conn.commit()
        conn.close()

        # Update in-memory data
        idx = self.known_face_ids.index(profile_id)
        self.known_face_encodings[idx].append((face_encoding, angle))

        return True

    def record_detection(self, profile_id, confidence, angle, image_path):
        """Record a face detection in the database_db"""
        conn = sqlite3.connect('face_profiles.db')
        c = conn.cursor()

        c.execute('''
        INSERT INTO detections (profile_id, timestamp, confidence, angle, image_path)
        VALUES (?, ?, ?, ?, ?)
        ''', (profile_id, datetime.now(), confidence, angle, image_path))

        # Update last seen
        c.execute('''
        UPDATE profiles
        SET last_seen = ?
        WHERE id = ?
        ''', (datetime.now(), profile_id))

        conn.commit()
        conn.close()

    def process_image(self, image_path, save_detections=True):
        """Process an image for face detection and recognition"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Detect faces
        faces = self.detect_faces(image)
        results = []

        for face in faces:
            # Get face bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Extract face
            face_img = image[y1:y2, x1:x2]

            # Estimate angle (simplified - in real implementation, you'd use pose estimation)
            angle = self.estimate_face_angle(face)

            # Get face encoding
            face_encoding = self.get_face_encoding(face_img, face)
            if face_encoding is None:
                continue

            # Match against known faces
            profile_id, name = self.match_face(face_encoding)

            if profile_id is None:
                # Unknown face
                result = {
                    'profile_id': None,
                    'name': 'Unknown',
                    'bbox': bbox,
                    'angle': angle,
                    'confidence': 0,
                    'face_img': face_img,
                    'face_info': face
                }
            else:
                # Known face
                result = {
                    'profile_id': profile_id,
                    'name': name,
                    'bbox': bbox,
                    'angle': angle,
                    'confidence': 1.0,  # Simplified - calculate actual confidence
                    'face_img': face_img,
                    'face_info': face
                }

                # Update database_db with new angle if significantly different
                if save_detections:
                    angles = [a for _, a in self.known_face_encodings[self.known_face_ids.index(profile_id)]]
                    if not any(abs(angle - a) < 15 for a in angles):
                        self.update_face_database(profile_id, face_img, face, angle)

                    # Record detection
                    self.record_detection(profile_id, result['confidence'], angle, image_path)

            results.append(result)

        return results

    def estimate_face_angle(self, face_info):
        """Estimate face angle based on landmarks (simplified)"""
        # In a real implementation, you would use the landmarks to calculate pose
        # For now, we'll return a random angle for demonstration
        return np.random.randint(-90, 90)

    def process_video(self, video_path, frame_interval=30, save_detections=True):
        """Process a video for face detection and recognition"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        all_results = []

        # Create output directory
        output_dir = Path("output_frames")
        output_dir.mkdir(exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Save frame
                frame_path = output_dir / f"frame_{frame_count}.jpg"
                cv2.imwrite(str(frame_path), frame)

                # Process frame
                results = self.process_image(str(frame_path), save_detections)
                if results:
                    all_results.extend(results)

        cap.release()
        return all_results


# Streamlit app
def main():
    parser = argparse.ArgumentParser(description="Face Recognition Security System WebApp")
    parser.add_argument('--db', type=str, default="database_db/face_profiles.db", help='Path to database file')
    parser.add_argument('--model_dir', type=str, default="models", help='Path to models directory')
    parser.add_argument('--window', type=str, default='Sentinel WebApp', help='Window name')
    args, unknown = parser.parse_known_args()

    st.set_page_config(
        page_title="Face Recognition Security System",
        page_icon="🔒",
        layout="wide"
    )

    st.title("Face Recognition Security System")
    st.markdown("A comprehensive system for detecting and recognizing faces in security footage")

    # Set model cache directories for InsightFace assets
    import os
    os.makedirs(args.model_dir, exist_ok=True)
    os.environ['INSIGHTFACE_HOME'] = args.model_dir
    os.environ['ONNX_HOME'] = args.model_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(args.model_dir, "huggingface")

    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state['db_manager'] = DatabaseManager(db_path=args.db)
    db = st.session_state['db_manager']

    # Initialize face detector and analyzer
    if 'face_detector' not in st.session_state:
        st.session_state['face_detector'] = FaceDetector()
    if 'face_analyzer' not in st.session_state:
        st.session_state['face_analyzer'] = FaceAnalyzer()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard",
        "Upload & Process",
        "Manage Profiles",
        "Analytics",
        "Live Detection",
        "Documentation"
    ])

    if page == "Dashboard":
        show_dashboard(db)
    elif page == "Upload & Process":
        show_upload_process(db, st.session_state['face_detector'], st.session_state['face_analyzer'])
    elif page == "Manage Profiles":
        show_manage_profiles_enhanced(db)
    elif page == "Analytics":
        show_analytics_enhanced(db)
    elif page == "Live Detection":
        show_live_detection(db, st.session_state['face_detector'])
    elif page == "Documentation":
        show_documentation()

    # Sidebar Danger Zone: Reset Database
    with st.sidebar.expander("Danger Zone", expanded=False):
        st.markdown("Resetting will delete the database file and clear caches. This cannot be undone.")
        also_delete_faces = st.checkbox("Also delete added_faces images", value=False)
        confirm = st.text_input("Type RESET to confirm", value="")
        if st.button("Reset Database", type="secondary"):
            if confirm.strip().upper() == "RESET":
                db_path = getattr(db, 'db_path', os.path.join("database_db", "face_profiles.db"))
                try:
                    if os.path.exists(db_path):
                        os.remove(db_path)
                    if also_delete_faces:
                        faces_dir = os.path.join(os.path.dirname(db_path), "added_faces")
                        if os.path.isdir(faces_dir):
                            shutil.rmtree(faces_dir, ignore_errors=True)
                    # Recreate DB manager and clear caches
                    st.session_state['db_manager'] = DatabaseManager(db_path=db_path)
                    for key in ['profiles_cache', 'known_face_encodings', 'known_face_ids']:
                        if key in st.session_state:
                            st.session_state.pop(key)
                    st.success("Database reset. Fresh database initialized.")
                except Exception as e:
                    st.error(f"Failed to reset database: {e}")
            else:
                st.warning("Confirmation text did not match. Type RESET to proceed.")


def show_dashboard(db):
    st.header("Dashboard")
    profiles = db.get_all_profiles()
    profiles_df = pd.DataFrame(profiles)
    st.subheader(f"Total Profiles: {len(profiles)}")

    # Recent detections with name mapping
    detections = db.get_detections(limit=200)
    detections_df = pd.DataFrame(detections)
    name_map = {p['id']: p.get('name', f"ID {p['id']}") for p in profiles}
    if not detections_df.empty:
        if 'profile_id' in detections_df.columns and 'name' not in detections_df.columns:
            detections_df['name'] = detections_df['profile_id'].map(name_map)
        st.write("Recent Detections:")
        st.dataframe(detections_df.tail(20))

        st.subheader("Detection Timeline")
        if 'timestamp' in detections_df.columns:
            detections_df['timestamp'] = pd.to_datetime(detections_df['timestamp'], errors='coerce')
            tl_df = detections_df.dropna(subset=['timestamp'])
            if not tl_df.empty:
                tl_df['date'] = tl_df['timestamp'].dt.date
                tl_df['name'] = tl_df['name'].fillna('Unknown')
                timeline_df = tl_df.groupby(['date', 'name']).size().reset_index(name='count')
                pivot_df = timeline_df.pivot(index='date', columns='name', values='count').fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                pivot_df.plot(kind='line', ax=ax, marker='o')
                plt.title('Detection Timeline')
                plt.xlabel('Date')
                plt.ylabel('Number of Detections')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

    # Quick demographics from profiles
    if not profiles_df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Gender Distribution")
            st.bar_chart(profiles_df['gender'].fillna('Unknown').value_counts())
        with c2:
            st.caption("Ethnicity Distribution")
            st.bar_chart(profiles_df['ethnicity'].fillna('Unknown').value_counts())
        with c3:
            st.caption("Age Range Distribution")
            st.bar_chart(profiles_df['age_range'].fillna('Unknown').value_counts())


def show_upload_process(db, face_detector, face_analyzer):
    st.header("Upload & Process Media")
    import pandas as pd
    import cv2
    upload_type = st.radio("Select upload type:", ["Image", "Video"])
    if upload_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file:
            file_bytes = uploaded_file.read()
            np_img = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            # Keep original in session for actions across reruns
            st.session_state['uploaded_image'] = image
            if st.button("Detect Faces"):
                ensure_db_cache(db)
                thresh = 0.2  # cosine similarity threshold (aligned with video_auto_enroll)
                faces = face_detector.process_frame(image)
                st.write(f"Detected {len(faces)} face(s)")
                # Draw overlays with matching and annotations
                overlay_img = image.copy()
                h, w = overlay_img.shape[:2]
                for i, face in enumerate(faces):
                    bx1, by1, bx2, by2 = [int(v) for v in face['bbox']]
                    # Clamp to frame
                    x1 = max(0, min(bx1, w-1))
                    y1 = max(0, min(by1, h-1))
                    x2 = max(0, min(bx2, w-1))
                    y2 = max(0, min(by2, h-1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    embedding = face.get('embedding')
                    match_found = False
                    matched_profile_id = None
                    label = ""
                    age = str(face.get('age', '-'))
                    gender = str(face.get('gender', '-'))
                    ethnicity = '-'

                    known_enc = st.session_state['known_face_encodings']
                    known_ids = st.session_state['known_face_ids']
                    if known_enc:
                        scores = batch_cosine_similarity(embedding, np.array(known_enc))
                        best_score = float(np.max(scores))
                        best_idx = int(np.argmax(scores))
                        if best_score > thresh:
                            match_found = True
                            matched_profile_id = known_ids[best_idx]

                    if match_found:
                        label = f"ID:{matched_profile_id}"
                        profile = next((p for p in st.session_state['profiles_cache'] if p['id'] == matched_profile_id), None)
                        db_gender = profile.get('gender', '-') if profile else '-'
                        db_age = profile.get('age_range', '-') if profile else '-'
                        db_eth = profile.get('ethnicity', '-') if profile else '-'
                        pose = face.get('pose')
                        yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                        label += f" | G:{db_gender} A:{db_age} E:{db_eth} | Pose:{yaw:.1f}"
                    else:
                        pose = face.get('pose')
                        yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                        label = f"Unknown | G:{gender} A:{age} E:{ethnicity} | Pose:{yaw:.1f}"

                    cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0,255,0) if match_found else (0,0,255), 2)
                    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    tx = max(0, x1)
                    ty = max(0, y1 - 10)
                    cv2.putText(overlay_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Persist results for add-to-DB actions after rerun
                st.session_state['detected_faces'] = faces
                st.session_state['detected_overlay'] = overlay_img
                st.success("Faces detected. You can now add faces to DB.")
            # Show the overlay if available; else show original
            if 'detected_overlay' in st.session_state:
                st.image(cv2.cvtColor(st.session_state['detected_overlay'], cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
                for i, face in enumerate(st.session_state.get('detected_faces', [])):
                    st.image(cv2.cvtColor(face['face_img'], cv2.COLOR_BGR2RGB), caption=f"Face {i+1}", width=100)
                    if st.button(f"Add Face {i+1} to DB", key=f"add_face_{i}"):
                        pid = db.add_profile(f"Unknown_{i+1}")
                        pose = face.get('pose')
                        yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                        db.add_face_encoding(pid, face['embedding'], yaw)
                        # Save snapshot and record detection
                        try:
                            db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
                            faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
                            os.makedirs(faces_dir, exist_ok=True)
                            img_path = os.path.join(faces_dir, f"face_{pid}.jpg")
                            cv2.imwrite(img_path, face['face_img'])
                            db.record_detection(pid, float(face.get('det_score', 0.0)), yaw, img_path)
                        except Exception as e:
                            st.warning(f"Could not save snapshot: {e}")
                        # refresh cache
                        st.session_state.pop('profiles_cache', None)
                        st.session_state.pop('known_face_encodings', None)
                        st.session_state.pop('known_face_ids', None)
                        ensure_db_cache(db)
                        st.success(f"Face {i+1} added as profile {pid}")
                    if st.button(f"Analyze Demographics {i+1}", key=f"analyze_face_{i}"):
                        attrs = face_analyzer.analyze_face(face['face_img'])
                        st.write(attrs)
            else:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
    else:
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"])
        if uploaded_file:
            import tempfile
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())
            temp_video.close()
            st.video(temp_video.name)
            auto_enroll = st.checkbox("Auto-enroll unknown faces during processing", value=True)
            if st.button("Process Video for Faces"):
                ensure_db_cache(db)
                cap = cv2.VideoCapture(temp_video.name)
                frame_count = 0
                detected_profiles = set()
                total_frames = int(max(1, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                progress = st.progress(0)
                preview_frame = st.empty()
                all_unknown_embeddings = []
                thresh = 0.2
                prev_time = time.time(); fps = 0.0
                fps_placeholder = st.empty()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % 30 != 0:
                        progress.progress(min(frame_count/total_frames, 1.0))
                        continue
                    faces = face_detector.process_frame(frame)
                    overlay_img = frame.copy()
                    ih, iw = overlay_img.shape[:2]
                    for i, face in enumerate(faces):
                        bx1, by1, bx2, by2 = [int(v) for v in face['bbox']]
                        x1 = max(0, min(bx1, iw-1))
                        y1 = max(0, min(by1, ih-1))
                        x2 = max(0, min(bx2, iw-1))
                        y2 = max(0, min(by2, ih-1))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        embedding = face.get('embedding')
                        match_found = False
                        matched_profile_id = None

                        known_enc = st.session_state['known_face_encodings']
                        known_ids = st.session_state['known_face_ids']
                        if known_enc:
                            scores = batch_cosine_similarity(embedding, np.array(known_enc))
                            best_score = float(np.max(scores))
                            best_idx = int(np.argmax(scores))
                            if best_score > thresh:
                                match_found = True
                                matched_profile_id = known_ids[best_idx]

                        # Demographics from detector for unknowns; DB for known
                        if match_found:
                            profile = next((p for p in st.session_state['profiles_cache'] if p['id'] == matched_profile_id), None)
                            db_gender = profile.get('gender', '-') if profile else '-'
                            db_age = profile.get('age_range', '-') if profile else '-'
                            db_eth = profile.get('ethnicity', '-') if profile else '-'
                            pose = face.get('pose')
                            yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                            label = f"ID:{matched_profile_id} | G:{db_gender} A:{db_age} E:{db_eth} | Pose:{yaw:.1f}"
                        else:
                            age = str(face.get('age', '-'))
                            gender = str(face.get('gender', '-'))
                            ethnicity = '-'
                            pose = face.get('pose')
                            yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                            label = f"Unknown | G:{gender} A:{age} E:{ethnicity} | Pose:{yaw:.1f}"

                        # Auto-enroll unknown if not duplicate
                        if auto_enroll and (not match_found) and not is_duplicate_unknown(embedding, all_unknown_embeddings, threshold=0.2):
                            pose = face.get('pose')
                            yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                            pid = db.add_profile(
                                name=f"Unknown_{frame_count}",
                                gender=str(gender),
                                age_range=str(age),
                                ethnicity=str(ethnicity),
                                notes=f"Auto-enrolled from video frame {frame_count}"
                            )
                            db.add_face_encoding(pid, embedding, angle=yaw)
                            # Save cropped snapshot and record detection
                            try:
                                db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
                                faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
                                os.makedirs(faces_dir, exist_ok=True)
                                img_path = os.path.join(faces_dir, f"face_{pid}.jpg")
                                crop = overlay_img[y1:y2, x1:x2]
                                if crop.size > 0:
                                    cv2.imwrite(img_path, crop)
                                db.record_detection(pid, float(face.get('det_score', 0.0)), yaw, img_path)
                            except Exception as e:
                                pass
                            # refresh cache with new profile
                            st.session_state.pop('profiles_cache', None)
                            st.session_state.pop('known_face_encodings', None)
                            st.session_state.pop('known_face_ids', None)
                            ensure_db_cache(db)
                            detected_profiles.add(pid)
                            all_unknown_embeddings.append(embedding)

                        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0,255,0) if match_found else (0,0,255), 2)
                        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        tx = max(0, x1)
                        ty = max(0, y1 - 10)
                        cv2.putText(overlay_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    # FPS
                    now = time.time(); dt = now - prev_time if prev_time else 0.001
                    fps = 0.9 * fps + 0.1 * (1.0/dt) if dt > 0 else fps
                    prev_time = now
                    fps_placeholder.markdown(f"**FPS:** {fps:.1f}")
                    preview_frame.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", use_container_width=True)
                    progress.progress(min(frame_count/total_frames, 1.0))
                cap.release()
                progress.empty()
                fps_placeholder.empty()
                preview_frame.empty()
                st.success(f"Processed video. Added {len(detected_profiles)} profiles.")

def show_manage_profiles_enhanced(db):
    st.header("Manage Profiles & Merge Duplicates")
    import pandas as pd
    profiles_df = pd.DataFrame(db.get_all_profiles())
    if profiles_df.empty:
        st.info("No profiles found.")
        return
    st.subheader("All Profiles")
    # Snapshots gallery per profile
    with st.expander("Show profile snapshots", expanded=False):
        cols = st.columns(4)
        # Preload detections once and build map pid->image_path
        try:
            dets = db.get_detections(limit=1000)
        except Exception:
            dets = []
        latest_img_by_pid = {}
        for det in dets:
            pid = det.get('profile_id')
            ip = det.get('image_path')
            if pid and ip and os.path.exists(ip) and pid not in latest_img_by_pid:
                latest_img_by_pid[pid] = ip
        # Base faces dir
        db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
        faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
        for idx, row in profiles_df.iterrows():
            pid = row['id']
            img_path = latest_img_by_pid.get(pid)
            if not img_path and os.path.isdir(faces_dir):
                candidate = os.path.join(faces_dir, f"face_{pid}.jpg")
                if os.path.exists(candidate):
                    img_path = candidate
            with cols[idx % 4]:
                st.markdown(f"**{row.get('name', 'Unknown')} (ID:{pid})**")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.caption("No snapshot available")
    search_term = st.text_input("Search profiles:")
    filtered_df = profiles_df[profiles_df['name'].str.contains(search_term, case=False)] if search_term else profiles_df
    # Use st.data_editor if available, else fallback to st.dataframe
    try:
        edited = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True)
    except AttributeError:
        st.warning("Streamlit version does not support st.data_editor. Displaying as read-only table.")
        edited = filtered_df
        st.dataframe(filtered_df)
    if st.button("Save Profile Edits"):
        for i, row in edited.iterrows():
            db.update_profile_notes(row['id'], row['notes'])
            db.update_profile_last_seen(row['id'])
        st.success("Profile edits saved.")
    st.subheader("Merge Profiles")
    merge_ids = st.multiselect("Select profiles to merge (choose 2+)", filtered_df['id'], format_func=lambda x: filtered_df[filtered_df['id']==x]['name'].values[0])
    if len(merge_ids) >= 2:
        mode = st.radio("Merge mode", ["Merge into existing target", "Create new merged profile"], horizontal=True)
        if mode == "Merge into existing target":
            target = st.selectbox("Target profile", merge_ids, format_func=lambda x: filtered_df[filtered_df['id']==x]['name'].values[0])
            sources = [i for i in merge_ids if i != target]
            if st.button("Merge into target"):
                merge_profiles_into(db, target, sources)
                st.success(f"Merged {sources} into {target}")
        else:
            new_name = st.text_input("Merged profile name", value="Merged Profile")
            if st.button("Create merged profile"):
                merge_profiles_db(db, merge_ids, new_name)
                st.success(f"Merged profiles into '{new_name}'.")
    st.subheader("Add New Profile")
    with st.form("add_profile_form"):
        new_name = st.text_input("Name")
        gender = st.selectbox("Gender", ["Unknown", "Male", "Female"])
        age_range = st.text_input("Age Range", value="Unknown")
        ethnicity = st.text_input("Ethnicity", value="Unknown")
        notes = st.text_area("Notes")
        uploaded = st.file_uploader("Upload face image (optional)", type=["jpg","jpeg","png","bmp","webp"]) 
        submitted = st.form_submit_button("Add Profile")
        if submitted and new_name:
            pid = db.add_profile(new_name, gender, age_range, ethnicity, notes)
            if uploaded:
                # Read and detect face, then save encoding + snapshot
                try:
                    file_bytes = uploaded.read()
                    np_img = np.frombuffer(file_bytes, np.uint8)
                    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    faces = st.session_state['face_detector'].process_frame(image)
                    if faces:
                        face = faces[0]
                        pose = face.get('pose')
                        yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                        db.add_face_encoding(pid, face['embedding'], yaw)
                        db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
                        faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
                        os.makedirs(faces_dir, exist_ok=True)
                        img_path = os.path.join(faces_dir, f"face_{pid}.jpg")
                        cv2.imwrite(img_path, face['face_img'])
                        db.record_detection(pid, float(face.get('det_score', 0.0)), yaw, img_path)
                except Exception as e:
                    st.warning(f"Could not process uploaded image: {e}")
            st.success(f"Added profile '{new_name}' (ID {pid}).")

    # Edit Profile fields directly
    st.subheader("Edit Profile")
    if not profiles_df.empty:
        pid_edit = st.selectbox("Select profile to edit", profiles_df['id'], format_func=lambda x: profiles_df[profiles_df['id']==x]['name'].values[0])
        prof = next((p for p in db.get_all_profiles() if p['id']==pid_edit), None)
        if prof:
            with st.form("edit_profile_form"):
                name_e = st.text_input("Name", value=prof.get('name',''))
                gender_e = st.selectbox("Gender", ["Unknown","Male","Female"], index=["Unknown","Male","Female"].index(prof.get('gender','Unknown') or 'Unknown'))
                age_e = st.text_input("Age Range", value=prof.get('age_range','Unknown') or 'Unknown')
                eth_e = st.text_input("Ethnicity", value=prof.get('ethnicity','Unknown') or 'Unknown')
                notes_e = st.text_area("Notes", value=prof.get('notes','') or '')
                submitted_e = st.form_submit_button("Save Changes")
                if submitted_e:
                    db.update_profile(pid_edit, name=name_e, gender=gender_e, age_range=age_e, ethnicity=eth_e, notes=notes_e)
                    st.success("Profile updated.")

def merge_profiles_db(db, profile_ids, merged_name):
    # Choose first profile as base for demographic fields
    base_profile = db.get_profile_by_id(profile_ids[0]) or {'gender': 'Unknown', 'age_range': 'Unknown', 'ethnicity': 'Unknown'}
    new_id = db.add_profile(merged_name, base_profile.get('gender', 'Unknown'), base_profile.get('age_range', 'Unknown'), base_profile.get('ethnicity', 'Unknown'), f"Merged from {profile_ids}")
    # Reassign data from all source profiles to new_id
    for pid in profile_ids:
        if pid == new_id:
            continue
        # Move encodings and detections
        db.reassign_profile_encodings(pid, new_id)
        db.reassign_profile_detections(pid, new_id)
        # Move/copy snapshot image if exists
        try:
            db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
            faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
            if os.path.isdir(faces_dir):
                src = os.path.join(faces_dir, f"face_{pid}.jpg")
                dst = os.path.join(faces_dir, f"face_{new_id}.jpg")
                if os.path.exists(src):
                    import shutil
                    shutil.copyfile(src, dst)
        except Exception:
            pass
        # Delete old profile to avoid duplicates
        db.delete_profile(pid)
    return new_id

def merge_profiles_into(db, target_profile_id, source_profile_ids):
    """Merge source profiles into an existing target profile, reassigning data and handling snapshots, then delete sources."""
    for pid in source_profile_ids:
        if pid == target_profile_id:
            continue
        db.reassign_profile_encodings(pid, target_profile_id)
        db.reassign_profile_detections(pid, target_profile_id)
        try:
            db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
            faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
            if os.path.isdir(faces_dir):
                src = os.path.join(faces_dir, f"face_{pid}.jpg")
                dst = os.path.join(faces_dir, f"face_{target_profile_id}.jpg")
                if os.path.exists(src) and not os.path.exists(dst):
                    import shutil
                    shutil.copyfile(src, dst)
        except Exception:
            pass
        db.delete_profile(pid)
    return target_profile_id

def show_analytics_enhanced(db):
    st.header("Analytics & Timeline Graph")
    import pandas as pd
    import plotly.express as px
    profiles = db.get_all_profiles()
    profiles_df = pd.DataFrame(profiles)
    detections = db.get_detections(limit=2000)
    detections_df = pd.DataFrame(detections)
    name_map = {p['id']: p.get('name', f"ID {p['id']}") for p in profiles}

    # Demographics from profiles always
    if not profiles_df.empty:
        st.subheader("Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Gender Distribution")
            st.bar_chart(profiles_df['gender'].fillna('Unknown').value_counts())
        with c2:
            st.caption("Ethnicity Distribution")
            st.bar_chart(profiles_df['ethnicity'].fillna('Unknown').value_counts())
        with c3:
            st.caption("Age Range Distribution")
            st.bar_chart(profiles_df['age_range'].fillna('Unknown').value_counts())

    # Timeline only if we have detections
    if not detections_df.empty:
        st.subheader("Profile Presence Timeline")
        if 'profile_id' in detections_df.columns and 'name' not in detections_df.columns:
            detections_df['name'] = detections_df['profile_id'].map(name_map)
        if 'timestamp' in detections_df.columns:
            detections_df['timestamp'] = pd.to_datetime(detections_df['timestamp'], errors='coerce')
            dd = detections_df.dropna(subset=['timestamp'])
            if not dd.empty:
                fig = px.scatter(
                    dd,
                    x='timestamp',
                    y='name',
                    color='name',
                    hover_data=['confidence', 'angle', 'image_path'] if set(['confidence','angle','image_path']).issubset(dd.columns) else None,
                    title="Profiles Present Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)


def show_live_detection(db, face_detector):
    st.header("Live Detection")
    st.info("This feature uses your webcam for real-time face detection. Press Start to begin.")
    import cv2
    import time
    if 'live_running' not in st.session_state:
        st.session_state['live_running'] = False

    cols = st.columns(2)
    if cols[0].button("Start Live Detection", key="live_start"):
        st.session_state['live_running'] = True
    if cols[1].button("Stop Live Detection", key="live_stop"):
        st.session_state['live_running'] = False

    stframe = st.empty()
    if st.session_state['live_running']:
        ensure_db_cache(db)
        cap = st.session_state.get('live_cap')
        if cap is None or not isinstance(cap, cv2.VideoCapture) or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            st.session_state['live_cap'] = cap
        ret, frame = cap.read()
        if ret:
            faces = face_detector.process_frame(frame)
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face['bbox']]
                # Clamp
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                # Match to DB
                label = ""
                embedding = face.get('embedding')
                match_found = False
                matched_profile_id = None
                known_enc = st.session_state['known_face_encodings']
                known_ids = st.session_state['known_face_ids']
                if known_enc:
                    scores = batch_cosine_similarity(embedding, np.array(known_enc))
                    best_score = float(np.max(scores))
                    best_idx = int(np.argmax(scores))
                    if best_score > 0.2:
                        match_found = True
                        matched_profile_id = known_ids[best_idx]
                if match_found:
                    profile = next((p for p in st.session_state['profiles_cache'] if p['id'] == matched_profile_id), None)
                    db_gender = profile.get('gender', '-') if profile else '-'
                    db_age = profile.get('age_range', '-') if profile else '-'
                    db_eth = profile.get('ethnicity', '-') if profile else '-'
                    pose = face.get('pose')
                    yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                    label = f"ID:{matched_profile_id} | G:{db_gender} A:{db_age} E:{db_eth} | Pose:{yaw:.1f}"
                else:
                    age = str(face.get('age', '-'))
                    gender = str(face.get('gender', '-'))
                    pose = face.get('pose')
                    yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                    label = f"Unknown | G:{gender} A:{age} | Pose:{yaw:.1f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0) if match_found else (0,0,255), 2)
                cv2.putText(frame, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            stframe.image(frame, channels="BGR")
        else:
            st.warning("Failed to read from camera.")
            st.session_state['live_running'] = False
        # Trigger rerun to keep loop going
        time.sleep(0.15)
        st.rerun()
    else:
        # Clean up camera if exists
        cap = st.session_state.get('live_cap')
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            st.session_state['live_cap'] = None
        stframe.empty()


def show_documentation():
    st.header("Documentation")
    docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'ARCHITECTURE.md')
    if os.path.exists(docs_path):
        try:
            with open(docs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Inline local images so they render inside Streamlit
            docs_dir = os.path.dirname(docs_path)

            def _to_data_uri(path: str) -> str:
                ext = os.path.splitext(path)[1].lower()
                mime = {
                    '.svg': 'image/svg+xml',
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp',
                }.get(ext, 'application/octet-stream')
                try:
                    with open(path, 'rb') as imgf:
                        b64 = base64.b64encode(imgf.read()).decode('ascii')
                    return f"data:{mime};base64,{b64}"
                except Exception:
                    return None

            # Replace markdown image syntax ![alt](path) and HTML <img src="path">
            import re
            def _replace_md_image(match):
                alt = match.group(1)
                src = match.group(2).strip()
                if src.startswith('http') or src.startswith('data:'):
                    return match.group(0)
                # Normalize relative path
                rel = src.lstrip('./').replace('/', os.sep)
                abspath = os.path.join(docs_dir, rel)
                if os.path.exists(abspath):
                    data_uri = _to_data_uri(abspath)
                    if data_uri:
                        return f'<img src="{data_uri}" alt="{alt}" />'
                return match.group(0)

            def _replace_html_img(match):
                src = match.group(1).strip()
                if src.startswith('http') or src.startswith('data:'):
                    return match.group(0)
                rel = src.lstrip('./').replace('/', os.sep)
                abspath = os.path.join(docs_dir, rel)
                if os.path.exists(abspath):
                    data_uri = _to_data_uri(abspath)
                    if data_uri:
                        # Replace only the src attribute value inside the tag
                        whole = match.group(0)
                        return whole.replace(src, data_uri)
                return match.group(0)

            # Apply replacements
            content = re.sub(r'!\[(.*?)\]\((.*?)\)', _replace_md_image, content)
            content = re.sub(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', _replace_html_img, content)

            st.markdown(content, unsafe_allow_html=True)
            st.caption("If images don't display, run Streamlit from the project root so relative paths like docs/assets/... resolve.")
        except Exception as e:
            st.error(f"Failed to load documentation: {e}")
    else:
        st.info("Documentation file not found. Expected at docs/ARCHITECTURE.md")

if __name__ == "__main__":
    main()