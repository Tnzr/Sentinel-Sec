# Sentinel-Sec Workshop (Python Conference Panama)

A beginner-friendly, follow-along tutorial built around this repository’s actual modules and app.

## What we’ll build
- Detect faces with `lib/face_detector.py` (InsightFace FaceAnalysis)
- Optionally analyze demographics with `lib/face_analyzer.py` (DeepFace)
- Store profiles/encodings/detections via `lib/database.py` (SQLite)
- Use cosine similarity to match faces (same logic as `src/sentinel.py` and `src/video_auto_enroll.py`)
- Run the Streamlit app (`src/sentinel.py`) and see results live

## Repo components we’ll use
- `lib/face_detector.py`
  - `FaceDetector.process_frame(image)` → list of faces dicts: `bbox, embedding, kps, pose, age, gender, det_score, angle, direction`
  - `visualize_detection(image, faces)` and `extract_faces_grid(image, faces)` for visuals
  - `create_test_image()` to generate a demo image
- `lib/face_analyzer.py`
  - `FaceAnalyzer.analyze_face(face_img)` → `{age, gender, ethnicity}`
- `lib/database.py`
  - `DatabaseManager.add_profile(name, gender, age_range, ethnicity, notes)`
  - `DatabaseManager.add_face_encoding(profile_id, embedding, angle)`
  - `DatabaseManager.record_detection(profile_id, confidence, angle, image_path)`
  - `DatabaseManager.get_all_profiles()`, `DatabaseManager.get_profile_encodings(pid)`
- `src/sentinel.py`
  - Cosine similarity threshold ≈ 0.2
  - Auto-enroll logic and overlays

## Environment prep (Windows PowerShell)
```powershell
# From repo root
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the workshop notebook
- Open `docs/workshop/pipeline_workshop.ipynb`
- Run top to bottom. It will:
  1. Check packages and GPU
  2. Set project paths and model cache dirs
  3. Initialize SQLite DB at `database_db/face_profiles.db`
  4. Detect faces on a synthetic image
  5. Visualize detections and faces grid
  6. Optionally run DeepFace demographics
  7. Persist a profile + encoding + snapshot + detection
  8. Demo cosine similarity matching vs encodings in DB
  9. Batch process images / sample a video and auto-enroll
  10. Plot detections and demographics
  11. Sanity checks
  12. Prepare artifacts for the webapp

## Matching logic (same as app)
```python
import numpy as np

def batch_cosine_similarity(embedding, embeddings):
    embedding = embedding.astype(np.float32)
    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    dot = np.dot(embeddings, embedding)
    norm_emb = np.linalg.norm(embeddings, axis=1)
    norm_query = np.linalg.norm(embedding)
    return dot / (norm_emb * norm_query + 1e-8)

THRESH = 0.2
```

## Start the Streamlit app
```powershell
# From repo root
python -m streamlit run src\sentinel.py
```

## Tips for beginners
- OpenCV is BGR, matplotlib expects RGB
- Clamp bbox within image bounds
- Use DeepFace sparingly during live/demo; enrich DB once per profile offline

## Troubleshooting
- If the app fails to start, install missing packages with `python -m pip install -r requirements.txt`
- If Documentation images don’t render, they are now inlined as data URIs
- If the DB looks empty, re-run the notebook sections for enrollment and detections
