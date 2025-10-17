"""
Face detection module using InsightFace with visualization and file selection
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional
from insightface.app import FaceAnalysis
from insightface.utils import ensure_available as _if_ensure_available
import traceback

# ===== SET CUSTOM MODEL PATHS =====
# Get the project root directory (Sentinel-Sec folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Set environment variables for model locations
os.environ['INSIGHTFACE_HOME'] = MODELS_DIR  # For InsightFace models
os.environ['DEEPFACE_HOME'] = MODELS_DIR  # For DeepFace models

# Also set for other common model directories
os.environ['ONNX_HOME'] = MODELS_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(MODELS_DIR, "huggingface")

print(f"Models will be saved to: {MODELS_DIR}")

# Import for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("tkinter not available. Using fallback file selection.")


class FaceDetector:
    """Detects faces in images and videos using InsightFace

    Parameters
    - model_name: insightface model name
    - device: 'auto'|'cpu'|'gpu' - choose runtime device. 'auto' will prefer GPU when available.
    """

    def __init__(self, model_name: str = 'buffalo_l', device: str = 'auto'):
        model_name = (model_name or 'buffalo_l').lower()
        if model_name != 'buffalo_l':
            # Only buffalo_l is supported in the optimized pipeline for stability.
            print(f"Model '{model_name}' is not supported; defaulting to 'buffalo_l'.")
            model_name = 'buffalo_l'

        # Determine ctx_id for InsightFace's FaceAnalysis.prepare
        # InsightFace expects ctx_id: -1 for CPU, 0..N for GPU device ids
        ctx_id = self._select_ctx_id(device)

        # Ensure the buffalo_l pack is present so we have detection + recognition modules.
        import glob
        try:
            pack_dir = _if_ensure_available('models', model_name, root=MODELS_DIR)
        except Exception as ee:
            raise RuntimeError(f"Failed to ensure model pack '{model_name}' at '{MODELS_DIR}': {ee}")
        files = sorted(glob.glob(os.path.join(pack_dir, '*.onnx')))
        if not files:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=f"insightface/{model_name}",
                    local_dir=pack_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=["*.onnx"],
                    resume_download=True
                )
                files = sorted(glob.glob(os.path.join(pack_dir, '*.onnx')))
            except Exception as dl_e:
                raise RuntimeError(
                    f"Unable to download InsightFace pack '{model_name}': {dl_e}. "
                    f"Add the ONNX files manually to '{pack_dir}'."
                )
        if not files:
            raise RuntimeError(f"InsightFace pack '{model_name}' does not contain ONNX models.")

        # Initialize the FaceAnalysis application with selected device
        # Explicitly pass providers when possible
        providers = None
        try:
            import onnxruntime as ort
            if ctx_id == -1:
                providers = ['CPUExecutionProvider']
            else:
                prov = ort.get_available_providers()
                if 'CUDAExecutionProvider' in prov:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
        except Exception:
            providers = None

        try:
            self.app = FaceAnalysis(
                name=model_name,
                providers=providers,
                root=MODELS_DIR,
                allowed_modules=['detection', 'recognition', 'genderage']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except Exception as inner_e:
            try:
                self.app = FaceAnalysis(
                    name=model_name,
                    providers=['CPUExecutionProvider'],
                    root=MODELS_DIR,
                    allowed_modules=['detection', 'recognition', 'genderage']
                )
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
            except Exception:
                tb = traceback.format_exc()
                raise RuntimeError(
                    f"InsightFace initialization failed for '{model_name}': {inner_e}\n{tb}"
                )

        self.model_name = model_name
        self.ctx_id = ctx_id
        self.has_display = self._check_display_support()

    @staticmethod
    def _pytorch_gpu_available() -> bool:
        """Return True if a CUDA-capable PyTorch is available."""
        try:
            import importlib
            torch_mod = importlib.import_module('torch')
            return bool(getattr(torch_mod, 'cuda', None) and torch_mod.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _onnxruntime_gpu_available() -> bool:
        """Return True if onnxruntime-gpu is installed and a CUDA provider is available."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            # Common GPU providers include 'CUDAExecutionProvider' and 'TensorrtExecutionProvider'
            return 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
        except Exception:
            return False

    def _select_ctx_id(self, device: str) -> int:
        """Select ctx_id for InsightFace based on requested device.

        Returns -1 for CPU, 0 for first GPU. If device='auto', prefer CUDA via PyTorch or ONNXRuntime.
        """
        device = (device or 'auto').lower()

        if device == 'cpu':
            return -1

        # If explicit GPU requested, try to validate presence
        if device == 'gpu' or device == 'auto':
            # Check common GPU-capable runtimes
            if self._pytorch_gpu_available() or self._onnxruntime_gpu_available():
                # Use GPU device 0 by default. If you need a different GPU, pass environment/INSIGHTFACE_CTX
                return 0

            # No GPU detected; fall back to CPU
            print("No GPU runtime detected. Falling back to CPU.")
            return -1

        # Unknown option, default to auto behavior
        return self._select_ctx_id('auto')

    def _check_display_support(self) -> bool:
        """Check if display is available"""
        try:
            # Try to create and destroy a simple window
            test_window = cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test")
            return True
        except:
            return False

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image

        Args:
            image: Input image in BGR format

        Returns:
            List of dictionaries containing face information
        """
        faces = self.app.get(image)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Ensure bbox is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 > x1 and y2 > y1:
                face_img = image[y1:y2, x1:x2]

                # Some models may lack certain attributes (age/gender/pose). Guard each access.
                try:
                    embedding = face.embedding
                except Exception:
                    embedding = None
                try:
                    kps = face.kps
                except Exception:
                    kps = None
                try:
                    pose = face.pose
                except Exception:
                    pose = None
                try:
                    age = int(face.age) if getattr(face, 'age', None) is not None else None
                except Exception:
                    age = None
                try:
                    gender = str(face.gender) if getattr(face, 'gender', None) is not None else None
                except Exception:
                    gender = None
                try:
                    det_score = float(face.det_score)
                except Exception:
                    det_score = 0.0

                result = {
                    'bbox': bbox,
                    'face_img': face_img,
                    'embedding': embedding,
                    'landmarks': kps,
                    'pose': pose,
                    'age': age,
                    'gender': gender,
                    'det_score': det_score
                }
                results.append(result)

        return results

    def estimate_face_angle(self, face_info: Dict) -> Tuple[float, str]:
        """
        Estimate face angle based on pose information

        Args:
            face_info: Face information dictionary

        Returns:
            Tuple of (angle in degrees, direction description)
        """
        angle = 0.0
        direction = "Frontal"

        if 'pose' in face_info and face_info['pose'] is not None:
            # Use pose estimation if available
            pose = face_info['pose']
            yaw, pitch, roll = pose

            # Determine direction based on yaw
            if yaw < -15:
                direction = "Looking Left"
            elif yaw > 15:
                direction = "Looking Right"
            elif pitch < -15:
                direction = "Looking Up"
            elif pitch > 15:
                direction = "Looking Down"
            else:
                direction = "Frontal"

            angle = yaw  # Use yaw as the primary angle

        # Fallback: estimate angle based on landmarks
        elif 'landmarks' in face_info and face_info['landmarks'] is not None:
            landmarks = face_info['landmarks']
            if len(landmarks) >= 2:
                # Use eye positions to estimate angle
                left_eye = landmarks[0]
                right_eye = landmarks[1]

                # Calculate angle between eyes
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))

                if angle < -5:
                    direction = "Tilted Left"
                elif angle > 5:
                    direction = "Tilted Right"
                else:
                    direction = "Straight"

        return angle, direction

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame for face detection

        Args:
            frame: Input frame in BGR format

        Returns:
            List of detected faces with additional information
        """
        faces = self.detect_faces(frame)

        for face in faces:
            angle, direction = self.estimate_face_angle(face)
            face['angle'] = angle
            face['direction'] = direction

        return faces

    def visualize_detection(self,
                          image: np.ndarray,
                          faces: List[Dict],
                          show_landmarks: bool = True,
                          show_angle: bool = True,
                          show_info: bool = True) -> np.ndarray:
        """
        Visualize face detections on the image

        Args:
            image: Original image
            faces: List of detected faces
            show_landmarks: Whether to show facial landmarks
            show_angle: Whether to show face angle
            show_info: Whether to show additional info

        Returns:
            Image with visualizations
        """
        result_image = image.copy()

        for i, face in enumerate(faces):
            bbox = face['bbox']
            landmarks = face.get('landmarks', [])
            angle = face.get('angle', 0)
            direction = face.get('direction', 'Unknown')
            det_score = face.get('det_score', 0)
            age = face.get('age', 0)
            gender = face.get('gender', 'Unknown')

            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw face number
            cv2.putText(result_image, f'Face {i+1}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw facial landmarks
            if show_landmarks and landmarks is not None:
                for j, landmark in enumerate(landmarks):
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(result_image, (x, y), 2, (0, 0, 255), -1)
                    cv2.putText(result_image, str(j), (x+3, y+3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Draw angle information
            if show_angle:
                angle_text = f'Angle: {angle:.1f}° - {direction}'
                cv2.putText(result_image, angle_text, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw additional information
            if show_info:
                info_y = y2 + 40
                info_lines = [
                    f'Score: {det_score:.3f}',
                    f'Age: {int(age)}',
                    f'Gender: {gender}'
                ]

                for line in info_lines:
                    cv2.putText(result_image, line, (x1, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    info_y += 15

        return result_image

    def extract_faces_grid(self,
                          image: np.ndarray,
                          faces: List[Dict],
                          face_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Extract detected faces and arrange them in a grid

        Args:
            image: Original image
            faces: List of detected faces
            face_size: Size to resize each face to

        Returns:
            Grid image with all detected faces
        """
        if not faces:
            # Return blank image if no faces detected
            return np.zeros((face_size[1], face_size[0], 3), dtype=np.uint8)

        face_images = []
        for i, face in enumerate(faces):
            face_img = face.get('face_img')
            if face_img is not None and face_img.size > 0:
                # Resize face to standard size
                resized_face = cv2.resize(face_img, face_size)

                # Add border and label
                bordered_face = cv2.copyMakeBorder(
                    resized_face, 30, 0, 5, 5, cv2.BORDER_CONSTANT, value=(50, 50, 50)
                )

                # Add face number and angle info
                angle = face.get('angle', 0)
                direction = face.get('direction', 'Unknown')
                info_text = f'Face {i+1}: {angle:.1f}°'
                cv2.putText(bordered_face, info_text, (5, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                face_images.append(bordered_face)

        if not face_images:
            return np.zeros((face_size[1], face_size[0], 3), dtype=np.uint8)

        # Arrange faces in a grid
        n_faces = len(face_images)
        cols = int(np.ceil(np.sqrt(n_faces)))
        rows = int(np.ceil(n_faces / cols))

        grid_height = rows * (face_size[1] + 30)
        grid_width = cols * (face_size[0] + 10)

        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for i, face_img in enumerate(face_images):
            row = i // cols
            col = i % cols
            y_start = row * (face_size[1] + 30)
            x_start = col * (face_size[0] + 10)
            grid_image[y_start:y_start+face_size[1]+30,
                      x_start:x_start+face_size[0]+10] = face_img

        return grid_image

    def save_visualizations(self,
                          image: np.ndarray,
                          faces: List[Dict],
                          output_dir: str = "output",
                          base_name: str = "detection") -> Dict[str, str]:
        """
        Save all visualizations to files without displaying

        Args:
            image: Original image
            faces: List of detected faces
            output_dir: Output directory
            base_name: Base name for output files

        Returns:
            Dictionary of saved file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Save detection visualization
        detection_viz = self.visualize_detection(image, faces)
        detection_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        cv2.imwrite(detection_path, detection_viz)
        saved_files['detection'] = detection_path

        # Save faces grid
        faces_grid = self.extract_faces_grid(image, faces)
        grid_path = os.path.join(output_dir, f"{base_name}_faces_grid.jpg")
        cv2.imwrite(grid_path, faces_grid)
        saved_files['faces_grid'] = grid_path

        # Save individual faces
        faces_dir = os.path.join(output_dir, f"individual_faces_{base_name}")
        os.makedirs(faces_dir, exist_ok=True)

        for i, face in enumerate(faces):
            face_img = face.get('face_img')
            if face_img is not None and face_img.size > 0:
                face_path = os.path.join(faces_dir, f"face_{i+1}.jpg")
                cv2.imwrite(face_path, face_img)
                saved_files[f'face_{i+1}'] = face_path

        return saved_files

    def show_face_preview(self, face: Dict, face_number: int, window_name: str = "Face Preview"):
        """
        Show a preview of the detected face in a separate window

        Args:
            face: Face information dictionary
            face_number: The number of the face in the sequence
            window_name: Name of the preview window
        """
        face_img = face.get('face_img')
        if face_img is not None and face_img.size > 0:
            # Resize for better display if too small
            h, w = face_img.shape[:2]
            if h < 100 or w < 100:
                scale = 300 / min(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                display_img = cv2.resize(face_img, (new_w, new_h))
            else:
                display_img = face_img.copy()

            # Add information overlay
            cv2.putText(display_img, f'Face {face_number}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            angle = face.get('angle', 0)
            direction = face.get('direction', 'Unknown')
            info_text = f'Angle: {angle:.1f}° - {direction}'
            cv2.putText(display_img, info_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow(window_name, display_img)
            cv2.waitKey(1)  # Refresh display


def select_image_file() -> Optional[str]:
    """
    Open a file dialog to select an image file

    Returns:
        Path to selected file or None if cancelled
    """
    if not HAS_TKINTER:
        # Fallback: use command line input
        print("\nFile dialog not available. Please enter image path manually:")
        print("Supported formats: JPG, JPEG, PNG, WEBP, BMP, TIFF")
        file_path = input("Enter image file path: ").strip().strip('"')
        if os.path.exists(file_path):
            return file_path
        else:
            print(f"File not found: {file_path}")
            return None

    # Create a hidden tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Set up file types
    file_types = [
        ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
        ("JPEG files", "*.jpg *.jpeg"),
        ("PNG files", "*.png"),
        ("WebP files", "*.webp"),
        ("All files", "*.*")
    ]

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image for face detection",
        filetypes=file_types,
        initialdir=os.getcwd()
    )

    # Destroy the hidden window
    root.destroy()

    return file_path if file_path else None


def create_test_image() -> np.ndarray:
    """Create a synthetic test image with faces"""
    height, width = 480, 640
    image = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)

    # Add some oval shapes to simulate faces
    face_colors = [(255, 200, 150), (200, 255, 200), (200, 200, 255)]
    face_positions = [(150, 120), (400, 150), (300, 300)]
    face_sizes = [(80, 100), (70, 90), (90, 110)]

    for color, (cx, cy), (w, h) in zip(face_colors, face_positions, face_sizes):
        # Draw face oval
        cv2.ellipse(image, (cx, cy), (w//2, h//2), 0, 0, 360, color, -1)

        # Draw eyes
        cv2.circle(image, (cx - w//4, cy - h//6), 5, (0, 0, 0), -1)
        cv2.circle(image, (cx + w//4, cy - h//6), 5, (0, 0, 0), -1)

        # Draw mouth
        cv2.ellipse(image, (cx, cy + h//4), (w//4, h//8), 0, 0, 180, (0, 0, 0), 2)

    return image


def main():
    """
    Standalone test for the FaceDetector.
    Detects faces and offers to add them to the database as new profiles.
    Shows face previews while processing.
    """
    from database import DatabaseManager
    from face_analyzer import FaceAnalyzer
    from utils import select_image_file

    print("Face Detection Module - Standalone Execution")
    print("=" * 50)

    # Initialize components
    detector = FaceDetector()
    db = DatabaseManager(db_path="../database_db/face_profiles.db")
    analyzer = FaceAnalyzer()

    # Select an image file
    image_path = select_image_file()
    if not image_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        sys.exit(1)

    # Show original image with detections first
    print(f"\nDetecting faces in {os.path.basename(image_path)}...")
    faces = detector.process_frame(image)

    if not faces:
        print("No faces detected in the image.")
        sys.exit(0)

    print(f"Found {len(faces)} face(s).\n")

    # Show the full image with all detections
    detection_image = detector.visualize_detection(image, faces)
    cv2.imshow('All Detected Faces - Press any key to continue', detection_image)
    print("Showing all detected faces. Press any key to continue to individual face processing...")
    cv2.waitKey(0)
    cv2.destroyWindow('All Detected Faces - Press any key to continue')

    # Process each detected face with preview
    for i, face in enumerate(faces):
        print(f"\n--- Face {i + 1} ---")
        bbox = face['bbox']
        angle = face.get('angle', 0)
        print(f"Bounding Box: {bbox}")
        print(f"Estimated Angle: {angle:.2f}°")

        # Show face preview
        detector.show_face_preview(face, i + 1, f"Face {i+1} Preview")

        # Ask user if they want to add this face to the database
        print("\nOptions:")
        print("y - Add this face to database")
        print("n - Skip this face")
        print("s - Show face details")
        print("q - Quit processing")

        choice = input("Choose option (y/n/s/q): ").strip().lower()

        if choice == 'q':
            print("Quitting...")
            break
        elif choice == 's':
            # Show more details
            print(f"Face details:")
            print(f"  Detection score: {face.get('det_score', 0):.3f}")
            print(f"  Estimated age: {face.get('age', 'Unknown')}")
            print(f"  Estimated gender: {face.get('gender', 'Unknown')}")
            print(f"  Direction: {face.get('direction', 'Unknown')}")
            # Show the face again
            detector.show_face_preview(face, i + 1, f"Face {i+1} Details")
            cv2.waitKey(0)
            # Ask again after showing details
            choice = input("Add this face to database? (y/n): ").strip().lower()

        if choice == 'y':
            # Get a name from the user
            name = input("Enter a name (or press Enter for a placeholder): ").strip()
            if not name:
                final_name = "Unknown_Person"
            else:
                final_name = name

            # Analyze face for demographic information
            print("Analyzing face for demographics...")
            try:
                demographics = analyzer.analyze_face(face['face_img'])
                print(f"Analysis results:")
                print(f"  Gender: {demographics.get('gender', 'Unknown')}")
                print(f"  Age: {demographics.get('age', 'Unknown')}")
                print(f"  Ethnicity: {demographics.get('ethnicity', 'Unknown')}")
            except Exception as e:
                print(f"Warning: Could not analyze demographics: {e}")
                demographics = {
                    'gender': 'Unknown',
                    'age': 'Unknown',
                    'ethnicity': 'Unknown'
                }

            # Add profile to database
            profile_id = db.add_profile(
                name=final_name,
                gender=demographics.get('gender'),
                age_range=demographics.get('age'),
                ethnicity=demographics.get('ethnicity'),
                notes="Added via face_detector __main__"
            )

            # If a placeholder was used, update it with the unique ID
            if not name:
                unique_name = f"Unknown_Person_{profile_id}"
                db.update_profile_notes(profile_id, f"Placeholder name: {unique_name}")
                print(f"✓ Profile created with placeholder name: '{unique_name}'")
            else:
                print(f"✓ Profile created with name: '{final_name}'")

            # Add face encoding to the profile
            embedding = face.get('embedding')
            if embedding is not None:
                db.add_face_encoding(profile_id, embedding, angle)
                print(f"✓ Face encoding and angle saved to profile ID: {profile_id}")
            else:
                print("⚠ Warning: Could not extract embedding for this face.")

            print(f"✓ Successfully saved Face {i+1} to database!")

        elif choice == 'n':
            print(f"⏭ Skipped Face {i+1}")
        else:
            print(f"⏭ Skipped Face {i+1}")

        # Close the preview window
        try:
            cv2.destroyWindow(f"Face {i+1} Preview")
        except Exception as e:
            pass

        print("-" * 50)

    # Close any remaining windows
    cv2.destroyAllWindows()
    print("\nProcessing completed!")


def inspect_database():
    """Inspect the database contents after processing"""
    from database import DatabaseManager

    db = DatabaseManager()

    print("\n" + "=" * 60)
    print("DATABASE INSPECTION")
    print("=" * 60)

    # Get all profiles
    profiles = db.get_all_profiles()
    print(f"\n📊 Total Profiles in Database: {len(profiles)}")

    for profile in profiles:
        print(f"\n👤 Profile ID: {profile['id']}")
        print(f"   Name: {profile['name']}")
        print(f"   Gender: {profile['gender']}")
        print(f"   Age Range: {profile['age_range']}")
        print(f"   Ethnicity: {profile['ethnicity']}")
        print(f"   Created: {profile['created_at']}")
        print(f"   Notes: {profile['notes']}")

        # Get face encodings for this profile
        encodings = db.get_profile_encodings(profile['id'])
        print(f"   📷 Face Encodings: {len(encodings)}")

        for encoding in encodings:
            # Database.get_profile_encodings currently returns a list of tuples: (embedding, angle)
            # Keep backward-compatible handling if it ever returns dict rows.
            if isinstance(encoding, tuple) or isinstance(encoding, list):
                emb, ang = encoding[0], encoding[1] if len(encoding) > 1 else None
                print(f"      - Encoding (tuple) Angle: {ang if ang is not None else 'Unknown'}")
                # Optionally show vector length for quick verification
                try:
                    print(f"         Vector length: {len(emb)}")
                except Exception:
                    pass
            elif isinstance(encoding, dict):
                # Older code path might return dict rows
                print(f"      - Encoding ID: {encoding.get('id', 'N/A')}, Angle: {encoding.get('angle', 'N/A')}")
            else:
                print(f"      - Encoding: {type(encoding)}")

    # Show statistics
    print(f"\n📈 DATABASE STATISTICS:")
    print(f"   Total profiles: {len(profiles)}")

    # Count by gender
    gender_counts = {}
    for profile in profiles:
        gender = profile['gender'] or 'Unknown'
        gender_counts[gender] = gender_counts.get(gender, 0) + 1

    for gender, count in gender_counts.items():
        print(f"   {gender}: {count}")

    return profiles

if __name__ == "__main__":
    main()
    inspect_database()