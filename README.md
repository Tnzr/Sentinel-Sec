# Sentinel-Sec

A Streamlit-based face recognition security system powered by InsightFace. It supports image and video processing, auto-enrollment of unknown faces, demographics enrichment, analytics, and live (webcam) detection.

## Features
- Streamlit web app (`src/sentinel.py`) with multiple tabs:
  - Dashboard: summary and timelines
  - Upload & Process: image/video detection, auto-enrollment, demographics
  - Manage Profiles: CRUD, merge duplicates, snapshots
  - Analytics: counts, timelines, demographics
  - Live Detection: webcam-based detection
  - Calibration: pick a good cosine threshold (WIP)
  - Documentation
- Cached model loading (prevents hangs on refresh)
- GPU/CPU switching via sidebar toggle
- SQLite database for profiles, encodings, detections
- Efficient cosine similarity matching and unknown deduplication

## Project structure
```
Sentinel-Sec/
├─ src/
│  ├─ sentinel.py             # Main Streamlit app
│  ├─ sentinel_chkpt.py       # Reference/previous checkpoint
│  ├─ video_auto_enroll.py    # Utilities for batch/video processing
│  └─ ...
├─ lib/
│  ├─ face_detector.py        # InsightFace wrapper (detector+embedder)
│  ├─ face_analyzer.py        # Demographics analyzer
│  ├─ database.py             # SQLite DB manager
│  └─ utils.py
├─ database_db/               # SQLite databases (ignored by git)
├─ models/                    # InsightFace models & caches (ignored by git)
├─ data/                      # Sample media (ignored by git)
├─ docs/                      # Architecture docs and assets
└─ README.md
```

## Requirements
- Python 3.9+ (3.10 recommended)
- OS: Linux/macOS/Windows
- Optional GPU: CUDA + onnxruntime-gpu (ensure only one ONNXRuntime wheel is installed)

Python packages (key):
- streamlit, numpy, opencv-contrib-python, onnxruntime or onnxruntime-gpu,
  insightface, pillow, pandas, matplotlib

System packages (Linux) if OpenCV fails to show images:
- `sudo apt-get install -y libgl1 libglib2.0-0`

Install dependencies:
- Prefer the generated list:
  - `pip install -r requirements_pipreqs.txt`
- If that’s not available/working:
  - `pip install -r requirements_backup.txt`

## Quickstart
1) Create and activate a virtual environment
- Linux/macOS
  - `python3 -m venv .venv && source .venv/bin/activate`
- Windows
  - `py -m venv .venv && .venv\\Scripts\\activate`

2) Install dependencies
- `pip install --upgrade pip`
- `pip install -r requirements_pipreqs.txt`  (or `requirements_backup.txt`)

3) Run the Streamlit app
- `streamlit run src/sentinel.py`

The app will download/load the InsightFace model (buffalo_l) on first run. Subsequent refreshes reuse the cached resources.

## Usage overview
- Sidebar
  - Matching Threshold slider (key: `cosine_threshold_v2`)
  - Force CPU toggle: disable GPU/providers if you face runtime/provider issues
  - Navigation: choose a page
  - Environment Info expander: shows ONNX providers, package versions, utilities
  - Danger Zone: reset DB and caches
- Database handling
  - Use Database Settings to switch or create DB files in `database_db/`
  - Snapshots stored under `added_faces/` beside the DB file
- Upload & Process
  - Image: detect faces, view overlays, add to DB, analyze demographics
  - Video: process video with configurable batch/display cadence
    - Auto-enroll unknown faces (with deduplication)
    - Demographics enrichment: Off, Post-run (recommended), Inline (slower)
- Manage Profiles
  - Edit fields, update notes/last seen
  - Merge duplicates (into target or into a new merged profile)
- Analytics
  - Demographics charts, detections per profile including zeros, timelines
- Live Detection
  - Webcam-based detection; start/stop buttons

## Performance and caching
- Heavy models are loaded via `st.cache_resource`:
  - FaceDetector: `get_face_detector(model_name="buffalo_l", device="auto")`
  - FaceAnalyzer: `get_face_analyzer(device="auto")`
- Session state stores instantiated resources to avoid duplicate widgets and frequent reinitialization on reruns.
- If providers/models get into a bad state, use "Reset InsightFace Cache" in the Environment Info expander and then refresh.

## Troubleshooting
- App hangs after refresh
  - Fixed via cached loaders in `src/sentinel.py`.
  - Try: Streamlit menu > Clear cache, then refresh.
  - Use Force CPU toggle to bypass GPU/provider issues.
- DuplicateWidgetID errors
  - Resolved by using unique keys (e.g., `cosine_threshold_v2`, `detections_to_load_slider`).
  - If encountered, clear cache and refresh.
- ONNXRuntime conflicts
  - Ensure only one of `onnxruntime` or `onnxruntime-gpu` is installed.
- OpenCV conflicts
  - Prefer only `opencv-contrib-python`. Uninstall extra `opencv-python(-headless)` wheels.
- Database errors
  - Ensure the `database_db/` directory is writable. Use Database Settings to switch DBs.

## Development
- Code is organized into `lib/` for core logic and `src/` for app entry points.
- Streamlit session-state is used to keep models and configuration stable across reruns.
- Contributions welcome. Open issues/PRs with clear reproduction steps and environment info.

## Preparing to push to GitHub
1) Initialize (already done) and ensure identity is set:
```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
2) Commit and push
```
git add .
git commit -m "Initial commit"
# Create a new empty repo on GitHub named Sentinel-Sec, then:
# Using SSH
git branch -M main
git remote add origin git@github.com:<your-user>/Sentinel-Sec.git
# Or using HTTPS
# git remote add origin https://github.com/<your-user>/Sentinel-Sec.git

git push -u origin main
```

`.gitignore` is configured to exclude models, databases, data samples, caches, and other artifacts.

## Notes
- InsightFace assets are cached under `models/`. The app sets `INSIGHTFACE_HOME`, `ONNX_HOME`, and `HUGGINGFACE_HUB_CACHE` to keep downloads local to the project.
- Calibration and some deeper analytics are WIP; placeholders exist in code.

## Contributing
See `CONTRIBUTING.md` for guidelines on reporting issues and submitting PRs.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
