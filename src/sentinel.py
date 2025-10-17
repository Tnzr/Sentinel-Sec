import os
import sys
# Set model cache directories for InsightFace assets at the very top
os.environ.setdefault('INSIGHTFACE_HOME', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
os.environ.setdefault('ONNX_HOME', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'huggingface'))

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
from uuid import uuid4

# Database management
import sqlite3
from pathlib import Path

# For video processing
import tempfile

from lib.database import DatabaseManager
from lib.face_detector import FaceDetector
from lib.face_analyzer import FaceAnalyzer

# --- CACHED MODEL LOADERS ---
@st.cache_resource(show_spinner="Loading FaceDetector...")
def get_face_detector(model_name="buffalo_l", device="auto"):
    return FaceDetector(model_name=model_name, device=device)

@st.cache_resource(show_spinner="Loading FaceAnalyzer...")
def get_face_analyzer(device="auto"):
    return FaceAnalyzer(device=device)

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


def _resolve_db_path(db_or_path) -> Path:
    if hasattr(db_or_path, 'db_path'):
        candidate = getattr(db_or_path, 'db_path', None)
    else:
        candidate = db_or_path
    if not candidate:
        candidate = os.path.join('database_db', 'face_profiles.db')
    return Path(candidate)


def _embedding_cache_path(db_or_path, ensure_dir: bool = False) -> Path:
    db_path = _resolve_db_path(db_or_path)
    cache_dir = db_path.parent / 'cache'
    if ensure_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{db_path.stem}_embeddings.npz"


def load_embedding_cache(db_or_path):
    cache_path = _embedding_cache_path(db_or_path)
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            encodings = [np.array(e, dtype=np.float32) for e in data['encodings']]
            ids = [int(i) for i in data['ids']]
            return encodings, ids
        except Exception:
            return None
    return None


def save_embedding_cache(db_or_path, encodings, ids):
    try:
        cache_path = _embedding_cache_path(db_or_path, ensure_dir=True)
        np.savez_compressed(
            cache_path,
            encodings=np.array(encodings, dtype=object),
            ids=np.array(ids, dtype=np.int64)
        )
    except Exception:
        pass


def invalidate_embedding_cache(db_or_path):
    try:
        cache_path = _embedding_cache_path(db_or_path)
        if cache_path.exists():
            cache_path.unlink()
    except Exception:
        pass


def clear_profile_cache(db):
    for key in ['profiles_cache', 'known_face_encodings', 'known_face_ids']:
        st.session_state.pop(key, None)
    invalidate_embedding_cache(db)


def ensure_db_cache(db):
    """Ensure profiles and encodings are cached in session_state."""
    if 'profiles_cache' not in st.session_state:
        st.session_state['profiles_cache'] = db.get_all_profiles()
    if 'known_face_encodings' not in st.session_state:
        st.session_state['known_face_encodings'] = []
    if 'known_face_ids' not in st.session_state:
        st.session_state['known_face_ids'] = []
    if not st.session_state['known_face_encodings'] or not st.session_state['known_face_ids']:
        cached = load_embedding_cache(db)
        if cached:
            encodings, ids = cached
            st.session_state['known_face_encodings'] = encodings
            st.session_state['known_face_ids'] = ids
        else:
            encodings_acc = []
            ids_acc = []
            for profile in st.session_state['profiles_cache']:
                encodings = db.get_profile_encodings(profile['id'])
                for emb, angle in encodings:
                    encodings_acc.append(emb.astype(np.float32))
                    ids_acc.append(profile['id'])
            st.session_state['known_face_encodings'] = encodings_acc
            st.session_state['known_face_ids'] = ids_acc
            if encodings_acc:
                save_embedding_cache(db, encodings_acc, ids_acc)
            else:
                invalidate_embedding_cache(db)

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Security System WebApp")
    parser.add_argument('--db', type=str, default="database_db/face_profiles.db", help='Path to database file')
    parser.add_argument('--model_dir', type=str, default="models", help='Path to models directory')
    parser.add_argument('--window', type=str, default='Sentinel WebApp', help='Window name')
    args, unknown = parser.parse_known_args()

    st.set_page_config(
        page_title="Face Recognition Security System",
        page_icon="👁️",
        layout="wide"
    )

    st.title("Face Recognition Security System")
    st.markdown("A comprehensive system for detecting and recognizing faces in security footage")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_db_path = args.db if os.path.isabs(args.db) else os.path.join(project_root, args.db)
    if 'db_path' not in st.session_state:
        st.session_state['db_path'] = default_db_path
    current_db_path = Path(st.session_state['db_path'])
    os.makedirs(current_db_path.parent, exist_ok=True)

    # --- Model selection ---
    st.sidebar.markdown("### Recognition Model")
    st.sidebar.info("Using InsightFace 'buffalo_l' for reliable GPU-accelerated inference.")
    model_choice = "buffalo_l"
    # Threshold slider (global) with single-instantiation guard
    st.sidebar.markdown("### Matching Threshold")
    if '_threshold_widget_created' not in st.session_state:
        thresh = st.sidebar.slider(
            "Cosine threshold",
            min_value=0.1,
            max_value=0.4,
            value=0.18,
            step=0.01,
            help="Higher = stricter match. Typical 0.30–0.40 for buffalo_l.",
            key="cosine_threshold_v2",
        )
        st.session_state['_threshold_widget_created'] = True
    else:
        thresh = st.session_state.get('cosine_threshold_v2', 0.18)
    # Toggle may not exist on older Streamlit; fallback to checkbox. Ensure unique keys.
    try:
        force_cpu = st.sidebar.toggle(
            "Force CPU (disable GPU)",
            value=st.session_state.get('force_cpu', False),
            help="Check to run models on CPU only. Useful for debugging provider issues.",
            key="force_cpu_toggle",
        )
    except AttributeError:
        force_cpu = st.sidebar.checkbox(
            "Force CPU (disable GPU)",
            value=st.session_state.get('force_cpu', False),
            key="force_cpu_checkbox",
        )
    st.session_state['force_cpu'] = force_cpu

    # Ensure session state keys are always initialized
    sidebar_error = None
    try:
        if 'db_manager' not in st.session_state:
            st.session_state['db_manager'] = DatabaseManager(db_path=st.session_state['db_path'])
        db = st.session_state['db_manager']
    except Exception as e:
        sidebar_error = f"Database initialization failed: {e}"
        db = None
    try:
        if 'face_detector' not in st.session_state or st.session_state.get('force_cpu') != force_cpu:
            device = 'cpu' if force_cpu else 'auto'
            st.session_state['face_detector'] = get_face_detector(model_name=model_choice, device=device)
            st.session_state['model_choice'] = model_choice
            st.session_state['force_cpu'] = force_cpu
    except Exception as e:
        sidebar_error = f"FaceDetector initialization failed: {e}"
    try:
        if 'face_analyzer' not in st.session_state or st.session_state.get('force_cpu') != force_cpu:
            device = 'cpu' if force_cpu else 'auto'
            st.session_state['face_analyzer'] = get_face_analyzer(device=device)
    except Exception as e:
        sidebar_error = f"FaceAnalyzer initialization failed: {e}"

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    if sidebar_error:
        st.sidebar.error(sidebar_error)
        st.sidebar.info("Try resetting cache or restarting the app.")
        # Still show tabs so user can access Danger Zone or Documentation
        page = st.sidebar.radio("Go to", ["Documentation", "Danger Zone"])
        if page == "Documentation":
            show_documentation()
        elif page == "Danger Zone":
            st.sidebar.markdown("Resetting will delete the database file and clear caches. This cannot be undone.")
            also_delete_faces = st.checkbox("Also delete added_faces images", value=False)
            confirm = st.text_input("Type RESET to confirm", value="")
            if st.button("Reset Database", type="secondary"):
                db_path = Path(st.session_state.get('db_path', os.path.join('database_db', 'face_profiles.db')))
                try:
                    invalidate_embedding_cache(db_path)
                    if db_path.exists():
                        db_path.unlink()
                    if also_delete_faces:
                        faces_dir = db_path.parent / 'added_faces'
                        if faces_dir.is_dir():
                            shutil.rmtree(faces_dir, ignore_errors=True)
                    st.session_state['db_path'] = str(db_path)
                    st.session_state.pop('db_manager', None)
                    st.success("Database reset. Please refresh the app.")
                except Exception as e:
                    st.error(f"Failed to reset database: {e}")
        return

    with st.sidebar.expander("Database Settings", expanded=False):
        st.caption(f"Current database: {current_db_path.name}")
        st.caption(str(current_db_path))
        existing_dbs = sorted(current_db_path.parent.glob('*.db'))
        existing_names = [p.name for p in existing_dbs]
        if existing_names:
            try:
                default_idx = existing_names.index(current_db_path.name)
            except ValueError:
                default_idx = 0
            selected_db = st.selectbox("Existing databases", existing_names, index=default_idx, key="existing_db_select")
            if st.button("Switch to selected", key="switch_db_btn"):
                new_path = current_db_path.parent / selected_db
                if str(new_path) != st.session_state['db_path']:
                    invalidate_embedding_cache(st.session_state['db_path'])
                    st.session_state['db_path'] = str(new_path)
                    st.session_state.pop('db_manager', None)
                    for cache_key in ['profiles_cache', 'known_face_encodings', 'known_face_ids']:
                        st.session_state.pop(cache_key, None)
                    invalidate_embedding_cache(new_path)
                    st.rerun()
        new_db_name = st.text_input("New database name", placeholder="custom_name.db", key="new_db_name")
        if st.button("Create or switch", key="create_db_btn"):
            candidate = new_db_name.strip()
            if not candidate:
                st.warning("Enter a database name.")
            else:
                if not candidate.lower().endswith('.db'):
                    candidate += '.db'
                new_path = current_db_path.parent / candidate
                invalidate_embedding_cache(st.session_state['db_path'])
                st.session_state['db_path'] = str(new_path)
                st.session_state.pop('db_manager', None)
                for cache_key in ['profiles_cache', 'known_face_encodings', 'known_face_ids']:
                    st.session_state.pop(cache_key, None)
                invalidate_embedding_cache(new_path)
                st.success(f"Using database {candidate}")
                st.rerun()

    prev_path = getattr(st.session_state.get('db_manager'), 'db_path', None)
    path_changed = bool(prev_path) and prev_path != st.session_state['db_path']
    if 'db_manager' not in st.session_state or path_changed:
        if path_changed:
            invalidate_embedding_cache(prev_path)
            for cache_key in ['profiles_cache', 'known_face_encodings', 'known_face_ids']:
                st.session_state.pop(cache_key, None)
        st.session_state['db_manager'] = DatabaseManager(db_path=st.session_state['db_path'])
        clear_profile_cache(st.session_state['db_manager'])
    db = st.session_state['db_manager']

    # --- Model config from session (avoid duplicate widgets) ---
    thresh = st.session_state.get('cosine_threshold_v2', 0.18)
    force_cpu = st.session_state.get('force_cpu', False)
    model_choice = st.session_state.get('model_choice', 'buffalo_l')

    # Ensure models exist without creating duplicate sliders/toggles
    if 'face_detector' not in st.session_state:
        try:
            device = 'cpu' if force_cpu else 'auto'
            st.session_state['face_detector'] = get_face_detector(model_name=model_choice, device=device)
        except Exception as e:
            st.sidebar.warning(f"FaceDetector init failed on {device}. Retrying CPU. Error: {e}")
            try:
                st.session_state['face_detector'] = get_face_detector(model_name='buffalo_l', device='cpu')
                st.session_state['model_choice'] = 'buffalo_l'
                st.session_state['force_cpu'] = True
            except Exception as e2:
                st.error(f"Could not initialize InsightFace: {e2}")
                return
    if 'face_analyzer' not in st.session_state:
        st.session_state['face_analyzer'] = get_face_analyzer(device=('cpu' if force_cpu else 'auto'))

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    with st.sidebar.expander("Environment Info", expanded=False):
        try:
            import onnxruntime as ort
            prov = ", ".join(ort.get_available_providers())
        except Exception:
            prov = "onnxruntime not available"
        # Avoid direct torch import to prevent errors when not installed
        cuda = False
        try:
            import importlib
            tmod = importlib.import_module('torch')
            if getattr(tmod, 'cuda', None):
                cuda = bool(tmod.cuda.is_available())
        except Exception:
            cuda = False
        st.caption(f"ONNX providers: {prov}")
        st.caption(f"PyTorch CUDA: {cuda}")

        # Package sanity checks to avoid ABI/provider issues
        try:
            from importlib import metadata as _meta
        except Exception:
            _meta = None
        def _pkg_version(name):
            if _meta is None:
                return None
            try:
                return _meta.version(name)
            except Exception:
                return None
        np_ver = getattr(np, '__version__', 'unknown')
        cv_ver = getattr(cv2, '__version__', 'unknown')
        st.caption(f"NumPy: {np_ver} | OpenCV: {cv_ver}")
        ort_cpu = _pkg_version('onnxruntime')
        ort_gpu = _pkg_version('onnxruntime-gpu')
        if ort_cpu and ort_gpu:
            st.warning(f"Both onnxruntime ({ort_cpu}) and onnxruntime-gpu ({ort_gpu}) are installed. Uninstall one to avoid conflicts.")
        elif ort_cpu:
            st.caption(f"onnxruntime (CPU): {ort_cpu}")
        elif ort_gpu:
            st.caption(f"onnxruntime-gpu (CUDA): {ort_gpu}")
        oc = _pkg_version('opencv-contrib-python')
        op = _pkg_version('opencv-python')
        oh = _pkg_version('opencv-python-headless')
        if op or oh:
            st.warning("Detected extra OpenCV wheels installed (opencv-python/headless). Keep only opencv-contrib-python to avoid conflicts.")
        else:
            if oc:
                st.caption(f"OpenCV contrib wheel: {oc}")

        # Utilities
        if st.button("Preload models (warm-up)"):
            try:
                t0 = time.time()
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = st.session_state['face_detector'].process_frame(dummy)
                # Optionally warm-up analyzer too (lazy init)
                crop = np.zeros((256, 256, 3), dtype=np.uint8)
                _ = st.session_state['face_analyzer'].analyze_face(crop)
                st.success(f"Models preloaded in {time.time()-t0:.1f}s")
            except Exception as e:
                st.warning(f"Preload failed: {e}")
        if st.button("Reset InsightFace Cache"):
            try:
                models_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
                # Common folders used by insightface under root
                purge = [os.path.join(models_root, '.insightface'), os.path.join(models_root, 'models')]
                removed = []
                for p in purge:
                    if os.path.exists(p):
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
                        removed.append(p)
                st.warning(f"Cleared: {removed}. Models will re-download on next initialization.")
                # Drop detector to force reinit
                for k in ['face_detector', 'model_choice']:
                    if k in st.session_state:
                        st.session_state.pop(k)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
    page = st.sidebar.radio("Go to", [
        "Dashboard",
        "Upload & Process",
        "Manage Profiles",
        "Analytics",
        "Live Detection",
        "Calibration",
        "Documentation"
    ])

    if page == "Dashboard":
        show_dashboard(db)
    elif page == "Upload & Process":
        show_upload_process(db, st.session_state['face_detector'], st.session_state['face_analyzer'], thresh)
    elif page == "Manage Profiles":
        show_manage_profiles_enhanced(db)
    elif page == "Analytics":
        show_analytics_enhanced(db)
    elif page == "Live Detection":
        show_live_detection(db, st.session_state['face_detector'], thresh)
    elif page == "Calibration":
        show_calibration(db)
    elif page == "Documentation":
        show_documentation()

    # Sidebar Danger Zone: Reset Database
    with st.sidebar.expander("Danger Zone", expanded=False):
        st.markdown("Resetting will delete the database file and clear caches. This cannot be undone.")
        also_delete_faces = st.checkbox("Also delete added_faces images", value=False)
        confirm = st.text_input("Type RESET to confirm", value="")
        if st.button("Reset Database", type="secondary"):
            if confirm.strip().upper() == "RESET":
                db_path = Path(st.session_state.get('db_path', default_db_path))
                try:
                    invalidate_embedding_cache(db_path)
                    if db_path.exists():
                        db_path.unlink()
                    if also_delete_faces:
                        faces_dir = db_path.parent / 'added_faces'
                        if faces_dir.is_dir():
                            shutil.rmtree(faces_dir, ignore_errors=True)
                    st.session_state['db_path'] = str(db_path)
                    st.session_state['db_manager'] = DatabaseManager(db_path=str(db_path))
                    clear_profile_cache(st.session_state['db_manager'])
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


def show_upload_process(db, face_detector, face_analyzer, thresh: float):
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
                    ethnicity = 'Unknown'

                    known_enc = st.session_state['known_face_encodings']
                    known_ids = st.session_state['known_face_ids']
                    if embedding is not None and known_enc:
                        try:
                            scores = batch_cosine_similarity(embedding, np.array(known_enc))
                            best_score = float(np.max(scores))
                            best_idx = int(np.argmax(scores))
                            if best_score > thresh:
                                match_found = True
                                matched_profile_id = known_ids[best_idx]
                        except Exception:
                            match_found = False

                    if match_found:
                        profile = next((p for p in st.session_state['profiles_cache'] if p['id'] == matched_profile_id), None)
                        db_gender = profile.get('gender', '-') if profile else '-'
                        db_age = profile.get('age_range', '-') if profile else '-'
                        db_eth = profile.get('ethnicity', '-') if profile else '-'
                        name_val = (profile.get('name', '') if profile else '') or ''
                        short_name = name_val
                        if short_name.lower().startswith('unknown_'):
                            short_name = short_name.split('_', 1)[1]
                        label = f"ID:{matched_profile_id} {short_name} | G:{db_gender} A:{db_age} E:{db_eth}"
                    else:
                        label = f"Unknown | G:{gender} A:{age} E:{ethnicity}"

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
                st.image(cv2.cvtColor(st.session_state['detected_overlay'], cv2.COLOR_BGR2RGB), caption="Processed Image", width='stretch')
                for i, face in enumerate(st.session_state.get('detected_faces', [])):
                    st.image(cv2.cvtColor(face['face_img'], cv2.COLOR_BGR2RGB), caption=f"Face {i+1}", width=100)
                    if st.button(f"Add Face {i+1} to DB", key=f"add_face_{i}"):
                        unique_tag = uuid4().hex[:8]
                        profile_name = f"Unknown_{unique_tag}"
                        pid = db.add_profile(profile_name)
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
                        clear_profile_cache(db)
                        ensure_db_cache(db)
                        st.success(f"Face {i+1} added as profile {pid} ({profile_name})")
                    if st.button(f"Analyze Demographics {i+1}", key=f"analyze_face_{i}"):
                        attrs = face_analyzer.analyze_face(face['face_img'])
                        st.write(attrs)
            else:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width='stretch')
    else:
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"], key="video_uploader")
        if uploaded_file:
            import tempfile
            video_token = f"{uploaded_file.name}-{uploaded_file.size}"
            previous_token = st.session_state.get('uploaded_video_token')
            if previous_token != video_token:
                old_path = st.session_state.pop('uploaded_video_path', None)
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except Exception:
                        pass
                suffix = os.path.splitext(uploaded_file.name)[1] or '.mp4'
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_video.write(uploaded_file.read())
                temp_video.close()
                st.session_state['uploaded_video_path'] = temp_video.name
                st.session_state['uploaded_video_token'] = video_token
            video_path = st.session_state.get('uploaded_video_path')
            if video_path and os.path.exists(video_path):
                st.video(video_path, width='stretch')

            if 'video_auto_enroll' not in st.session_state:
                st.session_state['video_auto_enroll'] = True
            if 'video_batch_size' not in st.session_state:
                st.session_state['video_batch_size'] = 15
            if 'video_display_every' not in st.session_state:
                st.session_state['video_display_every'] = 15

            st.checkbox("Auto-enroll unknown faces during processing", key="video_auto_enroll")
            st.slider("Frames per detection batch", min_value=1, max_value=30, step=1, key="video_batch_size")
            st.slider("Display every N frames", min_value=1, max_value=30, step=1, key="video_display_every")
            if 'video_attr_mode' not in st.session_state:
                st.session_state['video_attr_mode'] = "Post-run (recommended)"
            st.radio(
                "Demographics enrichment",
                ["Off", "Post-run (recommended)", "Inline (slower)"],
                key="video_attr_mode",
                help="Choose when to run InsightFace/FairFace age, gender, and ethnicity analysis for auto-enrolled faces."
            )

            if st.button("Process Video for Faces", key="process_video_btn"):
                st.session_state['processing_video'] = True

        if st.session_state.get('processing_video'):
            video_path = st.session_state.get('uploaded_video_path')
            batch_size = max(1, int(st.session_state.get('video_batch_size', 15)))
            display_every = max(1, int(st.session_state.get('video_display_every', 15)))
            auto_enroll = bool(st.session_state.get('video_auto_enroll', True))
            attr_mode = st.session_state.get('video_attr_mode', "Post-run (recommended)")
            if not video_path or not os.path.exists(video_path):
                st.error("Video file could not be found. Please upload again.")
                st.session_state['processing_video'] = False
            else:
                ensure_db_cache(db)
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                detected_profiles = set()
                total_frames = int(max(1, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                progress = st.progress(0)
                preview_frame = st.empty()
                all_unknown_embeddings = []
                prev_time = time.time(); fps = 0.0
                fps_placeholder = st.empty()
                pending_attrs = []
                inline_enriched = 0
                error_msg = None
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        overlay_img = frame.copy()
                        process_frame = (frame_count % batch_size) == 0 or frame_count == 1
                        if not process_frame:
                            # Show current frame even when skipping detection to avoid pauses
                            preview_frame.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", width='stretch')
                            progress.progress(min(frame_count/total_frames, 1.0))
                            # Update FPS
                            now = time.time(); dt = now - prev_time if prev_time else 0.001
                            fps = 0.9 * fps + 0.1 * (1.0/dt) if dt > 0 else fps
                            prev_time = now
                            fps_placeholder.markdown(f"**FPS:** {fps:.1f}")
                            continue
                        faces = face_detector.process_frame(frame)
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
                            if embedding is not None and known_enc:
                                try:
                                    scores = batch_cosine_similarity(embedding, np.array(known_enc))
                                    best_score = float(np.max(scores))
                                    best_idx = int(np.argmax(scores))
                                    if best_score > thresh:
                                        match_found = True
                                        matched_profile_id = known_ids[best_idx]
                                except Exception:
                                    match_found = False

                            if match_found:
                                profile = next((p for p in st.session_state['profiles_cache'] if p['id'] == matched_profile_id), None)
                                db_gender = profile.get('gender', '-') if profile else '-'
                                db_age = profile.get('age_range', '-') if profile else '-'
                                db_eth = profile.get('ethnicity', '-') if profile else '-'
                                name_val = (profile.get('name', '') if profile else '') or ''
                                short_name = name_val
                                if short_name.lower().startswith('unknown_'):
                                    short_name = short_name.split('_', 1)[1]
                                label = f"ID:{matched_profile_id} {short_name} | G:{db_gender} A:{db_age} E:{db_eth}"
                            else:
                                age = str(face.get('age', '-'))
                                gender = str(face.get('gender', '-'))
                                ethnicity = 'Unknown'
                                label = f"Unknown | G:{gender} A:{age} E:{ethnicity}"

                            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0,255,0) if match_found else (0,0,255), 2)
                            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            tx = max(0, x1)
                            ty = max(0, y1 - 10)
                            cv2.putText(overlay_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                            if auto_enroll and (not match_found) and (embedding is not None) and not is_duplicate_unknown(embedding, all_unknown_embeddings, threshold=0.2):
                                unique_tag = uuid4().hex[:8]
                                profile_name = f"Unknown_{unique_tag}"
                                pose = face.get('pose')
                                yaw = float(pose[0]) if (pose is not None and len(pose) > 0) else 0.0
                                pid = db.add_profile(
                                    name=profile_name,
                                    gender=str(gender),
                                    age_range=str(age),
                                    ethnicity=str(ethnicity),
                                    notes=f"Auto-enrolled as {profile_name} from video frame {frame_count}"
                                )
                                db.add_face_encoding(pid, embedding, angle=yaw)
                                try:
                                    db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
                                    faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
                                    os.makedirs(faces_dir, exist_ok=True)
                                    img_path = os.path.join(faces_dir, f"face_{pid}.jpg")
                                    # Use the original frame for the crop, not overlay_img
                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        cv2.imwrite(img_path, crop)
                                    db.record_detection(pid, float(face.get('det_score', 0.0)), yaw, img_path)
                                    if attr_mode == "Post-run (recommended)":
                                        pending_attrs.append((pid, crop.copy()))
                                    elif attr_mode == "Inline (slower)" and face_analyzer is not None:
                                        try:
                                            attrs = face_analyzer.analyze_face(crop)
                                            if attrs:
                                                db.update_profile(
                                                    pid,
                                                    gender=attrs.get('gender', 'Unknown'),
                                                    age_range=str(attrs.get('age', 'Unknown')),
                                                    ethnicity=attrs.get('ethnicity', 'Unknown')
                                                )
                                                inline_enriched += 1
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                clear_profile_cache(db)
                                ensure_db_cache(db)
                                detected_profiles.add(pid)
                                all_unknown_embeddings.append(embedding)

                        # Update FPS and UI every processed frame
                        now = time.time(); dt = now - prev_time if prev_time else 0.001
                        fps = 0.9 * fps + 0.1 * (1.0/dt) if dt > 0 else fps
                        prev_time = now
                        fps_placeholder.markdown(f"**FPS:** {fps:.1f}")
                        preview_frame.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", width='stretch')
                        progress.progress(min(frame_count/total_frames, 1.0))
                except Exception as proc_err:
                    error_msg = str(proc_err)
                finally:
                    cap.release()
                    progress.empty()
                    fps_placeholder.empty()
                    preview_frame.empty()
                    st.session_state['processing_video'] = False

                if error_msg:
                    st.error(f"Video processing failed: {error_msg}")
                else:
                    st.success(f"Processed video. Added {len(detected_profiles)} profiles.")
                    if attr_mode == "Post-run (recommended)" and pending_attrs and face_analyzer is not None:
                        enriched = 0
                        for pid, crop in pending_attrs:
                            try:
                                attrs = face_analyzer.analyze_face(crop)
                                if attrs:
                                    db.update_profile(
                                        pid,
                                        gender=attrs.get('gender', 'Unknown'),
                                        age_range=str(attrs.get('age', 'Unknown')),
                                        ethnicity=attrs.get('ethnicity', 'Unknown')
                                    )
                                    enriched += 1
                            except Exception:
                                continue
                        if enriched:
                            st.info(f"Updated demographics for {enriched} auto-enrolled profile(s) after processing.")
                    elif attr_mode == "Inline (slower)" and inline_enriched:
                        st.info(f"Inline demographics updated for {inline_enriched} auto-enrolled profile(s).")
                    elif attr_mode == "Off" and detected_profiles:
                        st.info("Skipped demographics enrichment per video settings.")

def show_manage_profiles_enhanced(db):
    st.header("Manage Profiles & Merge Duplicates")
    import pandas as pd
    raw_profiles = db.get_all_profiles()
    profiles_df = pd.DataFrame(raw_profiles)
    if profiles_df.empty:
        st.info("No profiles in the selected database yet. Use the form below to add the first profile.")
        profiles_df = pd.DataFrame(columns=['id','name','gender','age_range','ethnicity','notes'])
    if not profiles_df.empty:
        st.subheader("All Profiles")
        with st.expander("Show profile snapshots", expanded=False):
            cols = st.columns(4)
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
            db_path = getattr(db, 'db_path', os.path.join('database_db', 'face_profiles.db'))
            faces_dir = os.path.join(os.path.dirname(db_path), 'added_faces')
            for idx, row in profiles_df.iterrows():
                pid = row.get('id')
                img_path = latest_img_by_pid.get(pid)
                if not img_path and faces_dir and os.path.isdir(faces_dir):
                    candidate = os.path.join(faces_dir, f"face_{pid}.jpg")
                    if os.path.exists(candidate):
                        img_path = candidate
                with cols[idx % 4]:
                    st.markdown(f"**{row.get('name', 'Unknown')} (ID:{pid})**")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.caption("No snapshot available")
    if not profiles_df.empty:
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
            st.session_state.pop('profiles_cache', None)
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
                    clear_profile_cache(db)
                    ensure_db_cache(db)
                    st.success(f"Merged {sources} into {target}")
            else:
                new_name = st.text_input("Merged profile name", value="Merged Profile")
                if st.button("Create merged profile"):
                    merge_profiles_db(db, merge_ids, new_name)
                    clear_profile_cache(db)
                    ensure_db_cache(db)
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
            clear_profile_cache(db)
            ensure_db_cache(db)
            st.success(f"Added profile '{new_name}' (ID {pid}).")

    # Edit Profile fields directly
    st.subheader("Edit Profile")
    if not profiles_df.empty:
        pid_edit = st.selectbox("Select profile to edit", profiles_df['id'], format_func=lambda x: profiles_df[profiles_df['id']==x]['name'].values[0])
        prof = next((p for p in db.get_all_profiles() if p['id']==pid_edit), None)
        if prof:
            genders = ["Unknown","Male","Female"]
            try:
                gender_index = genders.index(prof.get('gender','Unknown') or 'Unknown')
            except ValueError:
                gender_index = 0
            with st.form("edit_profile_form"):
                name_e = st.text_input("Name", value=prof.get('name',''))
                gender_e = st.selectbox("Gender", genders, index=gender_index)
                age_e = st.text_input("Age Range", value=prof.get('age_range','Unknown') or 'Unknown')
                eth_e = st.text_input("Ethnicity", value=prof.get('ethnicity','Unknown') or 'Unknown')
                notes_e = st.text_area("Notes", value=prof.get('notes','') or '')
                submitted_e = st.form_submit_button("Save Changes")
                if submitted_e:
                    db.update_profile(pid_edit, name=name_e, gender=gender_e, age_range=age_e, ethnicity=eth_e, notes=notes_e)
                    st.session_state.pop('profiles_cache', None)
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
    invalidate_embedding_cache(db)
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
    invalidate_embedding_cache(db)
    return target_profile_id

def show_analytics_enhanced(db):
    st.header("Analytics & Timeline Graph")
    import pandas as pd
    import plotly.express as px
    profiles = db.get_all_profiles()
    profiles_df = pd.DataFrame(profiles)
    # Allow a larger pull to capture history
    limit = st.slider("Detections to load", min_value=500, max_value=10000, value=2000, step=500, key="detections_to_load_slider")
    detections = db.get_detections(limit=int(limit))
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

    # Presence counts including profiles with zero detections in current window
    st.subheader("Detections per Profile (including zero)")
    if 'id' in profiles_df.columns:
        counts = pd.DataFrame(columns=['profile_id','count'])
        if not detections_df.empty and 'profile_id' in detections_df.columns:
            counts = detections_df.groupby('profile_id').size().reset_index(name='count')
        merged = profiles_df.rename(columns={'id':'profile_id'})[['profile_id','name']].merge(counts, on='profile_id', how='left')
        merged['count'] = merged['count'].fillna(0).astype(int)
        # Fallback name when missing
        merged['name'] = merged.apply(lambda r: r['name'] if pd.notna(r['name']) and str(r['name']).strip() else f"ID {r['profile_id']}", axis=1)
        st.dataframe(merged.sort_values('count', ascending=False), use_container_width=True)

    # Timeline only if we have detections
    if not detections_df.empty:
        st.subheader("Profile Presence Timeline")
        if 'profile_id' in detections_df.columns and 'name' not in detections_df.columns:
            detections_df['name'] = detections_df['profile_id'].map(name_map)
            # Ensure every profile_id has a label even if profile was deleted or missing
            detections_df['name'] = detections_df.apply(
                lambda r: r['name'] if pd.notna(r['name']) and str(r['name']).strip() else f"ID {r['profile_id']}", axis=1
            )
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


def show_live_detection(db, face_detector, thresh: float):
    st.header("Live Detection")
    st.info("This feature uses your webcam for real-time face detection. Press Start to begin.")
    import cv2
    import time
    # Auto-start when opening the tab
    if 'live_running' not in st.session_state:
        st.session_state['live_running'] = True

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
            try:
                # Try multiple backends
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(0)
            except Exception:
                cap = cv2.VideoCapture(0)
            st.session_state['live_cap'] = cap
        # Tiny delay before first read can help some backends
        time.sleep(0.15)
        # Retry reads a few times before failing
        ret, frame = cap.read()
        if not ret:
            for _ in range(10):
                time.sleep(0.05)
                ret, frame = cap.read()
                if ret:
                    break
        if ret:
            faces = face_detector.process_frame(frame)
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face['bbox']]
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
                if embedding is not None and known_enc:
                    try:
                        scores = batch_cosine_similarity(embedding, np.array(known_enc))
                        best_score = float(np.max(scores))
                        best_idx = int(np.argmax(scores))
                        if best_score > thresh:
                            match_found = True
                            matched_profile_id = known_ids[best_idx]
                    except Exception:
                        match_found = False
                if match_found:
                    profile = next((p for p in st.session_state['profiles_cache'] if p['id'] == matched_profile_id), None)
                    db_gender = profile.get('gender', '-') if profile else '-'
                    db_age = profile.get('age_range', '-') if profile else '-'
                    db_eth = profile.get('ethnicity', '-') if profile else '-'
                    name_val = (profile.get('name', '') if profile else '') or ''
                    short_name = name_val
                    if short_name.lower().startswith('unknown_'):
                        short_name = short_name.split('_', 1)[1]
                    label = f"ID:{matched_profile_id} {short_name} | G:{db_gender} A:{db_age} E:{db_eth}"
                else:
                    age = str(face.get('age', '-'))
                    gender = str(face.get('gender', '-'))
                    label = f"Unknown | G:{gender} A:{age}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0) if match_found else (0,0,255), 2)
                cv2.putText(frame, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            stframe.image(frame, channels="BGR")
        else:
            st.warning("Failed to read from camera.")
            st.session_state['live_running'] = False
        time.sleep(0.15)
        st.rerun()
    else:
        cap = st.session_state.get('live_cap')
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            st.session_state['live_cap'] = None
        stframe.empty()

def show_calibration(db):
    st.header("Calibration: Score Distributions")
    st.info("This helps pick a good cosine threshold. We'll draw distributions for genuine vs. impostor pairs from your current DB.")
    profiles = db.get_all_profiles()
    # Build embeddings by profile
    by_pid = {}
    for p in profiles:
        encs = db.get_profile_encodings(p['id'])
        if encs:
            by_pid[p['id']] = [emb.astype(np.float32) for emb, ang in encs]
    # Compute cosine scores
    genuine = []
    impostor = []
    def cos(a,b):
        a = a.astype(np.float32); b = b.astype(np.float32)
        return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))
    # Genuine: within same profile
    for pid, embs in by_pid.items():
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                genuine.append(cos(embs[i], embs[j]))
    # Impostor: across different profiles (sampled)
    pids = list(by_pid.keys())
    for i in range(len(pids)):
        for j in range(i+1, len(pids)):
            Ei = by_pid[pids[i]]; Ej = by_pid[pids[j]]
            if Ei and Ej:
                impostor.append(cos(Ei[0], Ej[0]))
    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4))
    if genuine:
        ax.hist(genuine, bins=30, alpha=0.6, color='green', label='Genuine')
    if impostor:
        ax.hist(impostor, bins=30, alpha=0.6, color='red', label='Impostor')
    ax.set_title('Cosine score distributions')
    ax.set_xlabel('Cosine score')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)
    st.caption("Aim for a threshold that lies between the two distributions, minimizing overlap.")


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