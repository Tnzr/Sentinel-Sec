"""Face analysis utilities powered by InsightFace and optional FairFace race classifier."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - onnxruntime should be available alongside InsightFace
    ort = None

# Remove top-level torch import; detect at runtime to avoid linter errors
import importlib

torch = None  # set when available at runtime

try:
    from insightface.app import FaceAnalysis
except Exception as exc:  # pragma: no cover - InsightFace is required
    raise ImportError("InsightFace is required for FaceAnalyzer") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Keep InsightFace aligned with the rest of the project
os.environ.setdefault("INSIGHTFACE_HOME", str(MODELS_DIR))
os.environ.setdefault("ONNX_HOME", str(MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR / "huggingface"))


class FaceAnalyzer:
    """Run lightweight demographic analysis (age, gender, ethnicity).

    Age and gender predictions come directly from InsightFace's genderage head.
    Ethnicity is estimated via a FairFace ONNX classifier when available; otherwise
    it falls back to 'Unknown'.
    """

    _RACE_MODEL_URL = (
        "https://huggingface.co/serengil/deepface_models/resolve/main/fairface_race.onnx?download=1"
    )
    # Default label order used by common FairFace ONNX exports (7 classes)
    _RACE_LABELS = [
        "White",
        "Black",
        "Latino/Hispanic",
        "East Asian",
        "Southeast Asian",
        "Indian",
        "Middle Eastern",
    ]

    def __init__(self, device: str = "auto") -> None:
        self.actions = ["age", "gender", "race"]
        # Defer heavy initialization; compute context now, but sessions later
        self._requested_device = (device or "auto").lower()
        self._ctx_id = self._select_ctx_id(self._requested_device)
        self._providers = self._select_providers(self._ctx_id)
        self._insight = None  # type: Optional[FaceAnalysis]
        self._race_session = None
        self._race_input_layout: Optional[str] = None  # 'NCHW' or 'NHWC'
        self._race_input_name: Optional[str] = None
        self._race_confidence_threshold: float = 0.1

    @staticmethod
    def _select_ctx_id(device: str) -> int:
        """Return InsightFace ctx_id: -1 (CPU) or >=0 (GPU)."""

        requested = (device or "auto").lower()
        if requested == "cpu":
            return -1

        if requested in {"gpu", "auto"}:
            # Try PyTorch CUDA
            try:
                torch_mod = importlib.import_module("torch")
                if getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available():
                    return 0
            except Exception:
                pass
            # Try ONNX Runtime CUDA/TensorRT
            if ort is not None:
                try:
                    providers = ort.get_available_providers()
                except Exception:
                    providers = []
                if any(p.startswith("CUDA") or p.startswith("Tensorrt") for p in providers):
                    return 0
        return -1

    @staticmethod
    def _select_providers(ctx_id: int) -> Optional[List[str]]:
        if ort is None:
            return None
        try:
            providers = ort.get_available_providers()
        except Exception:
            return None
        if ctx_id == -1:
            return ["CPUExecutionProvider"] if "CPUExecutionProvider" in providers else None
        if "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in providers else None

    @staticmethod
    def _load_insight(ctx_id: int, providers: Optional[List[str]]) -> FaceAnalysis:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR),
            providers=providers,
            allowed_modules=["detection", "genderage"],
        )
        app.prepare(ctx_id=ctx_id, det_size=(256, 256))
        return app

    def _load_race_classifier(self, providers: Optional[List[str]]):
        if ort is None:
            return None
        model_path = MODELS_DIR / "demography" / "fairface_race.onnx"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(self._RACE_MODEL_URL, model_path)
            except Exception:
                return None
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            # Cache input meta
            inp = session.get_inputs()[0]
            self._race_input_name = inp.name
            shape = list(inp.shape)
            # Determine layout by channel index
            if len(shape) == 4:
                if shape[1] == 3:
                    self._race_input_layout = 'NCHW'
                elif shape[-1] == 3:
                    self._race_input_layout = 'NHWC'
                else:
                    # Fallback; most exports are NCHW
                    self._race_input_layout = 'NCHW'
            else:
                self._race_input_layout = 'NCHW'
        except Exception:
            session = None
        return session

    def _ensure_initialized(self) -> None:
        """Lazily construct heavy models/sessions on first use."""
        if self._insight is None:
            self._insight = self._load_insight(self._ctx_id, self._providers)
        if self._race_session is None:
            self._race_session = self._load_race_classifier(self._providers)

    @staticmethod
    def _normalize_gender(raw: Optional[float]) -> str:
        if raw is None:
            return "Unknown"
        if isinstance(raw, (int, float)):
            return "Female" if raw > 0.5 else "Male"
        if isinstance(raw, str):
            lowered = raw.lower()
            if lowered.startswith("f"):
                return "Female"
            if lowered.startswith("m"):
                return "Male"
        return "Unknown"

    @staticmethod
    def _imagenet_normalize(image: np.ndarray) -> np.ndarray:
        # Expect image in RGB, float32 in [0,1]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (image - mean) / std

    @staticmethod
    def _half_normalize(image: np.ndarray) -> np.ndarray:
        # Normalize to [-1,1] as used by some FairFace exports
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return (image - mean) / std

    def _prep_race_input(self, face_img: np.ndarray, layout: str, strategy: str) -> np.ndarray:
        # Convert BGR -> RGB, resize to 224
        image = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        if strategy == 'imagenet':
            image = self._imagenet_normalize(image)
        else:
            image = self._half_normalize(image)
        if layout == 'NCHW':
            image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, ...].astype(np.float32)
        return image

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def _predict_race(self, face_img: np.ndarray) -> str:
        if self._race_session is None or self._race_input_name is None or self._race_input_layout is None:
            return "Unknown"
        try:
            # Try two common normalization strategies
            for strategy in ('imagenet', 'half'):
                inp = self._prep_race_input(face_img, self._race_input_layout, strategy)
                outputs = self._race_session.run(None, {self._race_input_name: inp})[0]
                if outputs.ndim == 2:
                    outputs = outputs[0]
                probs = self._softmax(outputs)
                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])
                if len(self._RACE_LABELS) == probs.shape[0]:
                    label = self._RACE_LABELS[best_idx]
                else:
                    # Unknown export label order/size; just report index
                    label = f"Class_{best_idx}"
                if best_prob >= self._race_confidence_threshold:
                    return label
            # Low-confidence after both strategies
            return "Unknown"
        except Exception:
            return "Unknown"

    def analyze_face(self, face_img: np.ndarray) -> Dict[str, str]:
        try:
            self._ensure_initialized()
            faces = self._insight.get(face_img)
            if not faces:
                raise ValueError("No face detected in provided crop")
            face = faces[0]
            age_val = getattr(face, "age", None)
            gender_val = getattr(face, "gender", None)
            age = str(int(round(age_val))) if isinstance(age_val, (int, float)) else "Unknown"
            gender = self._normalize_gender(gender_val)
            ethnicity = self._predict_race(face_img)
            return {
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
            }
        except Exception:
            return {"age": "Unknown", "gender": "Unknown", "ethnicity": "Unknown"}

    def batch_analyze(self, face_images: List[np.ndarray]) -> List[Dict[str, str]]:
        self._ensure_initialized()
        return [self.analyze_face(img) for img in face_images]


if __name__ == "__main__":
    from face_detector import FaceDetector
    from utils import select_image_file

    print("Face Analysis Module - Standalone Execution")
    print("=" * 50)

    image_path = select_image_file()
    if not image_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    detector = FaceDetector()
    analyzer = FaceAnalyzer()

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        sys.exit(1)

    print(f"\nDetecting faces in {os.path.basename(image_path)}...")
    faces = detector.process_frame(image)
    if not faces:
        print("No faces detected in the image.")
        sys.exit(0)

    print(f"Found {len(faces)} face(s). Analyzing...\n")
    for idx, face in enumerate(faces, start=1):
        attrs = analyzer.analyze_face(face['face_img'])
        print(f"--- Face {idx} ---")
        for key, value in attrs.items():
            print(f"  {key.capitalize()}: {value}")
        print("-" * 20)