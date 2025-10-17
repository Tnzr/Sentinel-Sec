"""
Face recognition module for matching faces against known database_db
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from .database import DatabaseManager


class FaceRecognizer:
    """Recognizes faces by comparing against known encodings"""

    def __init__(self, db_manager: DatabaseManager, threshold: float = 0.6):
        self.db_manager = db_manager
        self.threshold = threshold
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from database_db"""
        profiles = self.db_manager.get_all_profiles()

        for profile in profiles:
            profile_id = profile['id']
            name = profile['name']
            encodings = [
                (np.asarray(emb, dtype=np.float32), angle)
                for emb, angle in self.db_manager.get_profile_encodings(profile_id)
            ]

            if encodings:
                self.known_face_ids.append(profile_id)
                self.known_face_names.append(name)
                self.known_face_encodings.append(encodings)
                # Debug: print encoding shape and sample value
                for known_encoding, angle in encodings:
                    print(f"Loaded encoding for profile {profile_id} (name={name}): shape={getattr(known_encoding, 'shape', None)}, angle={angle}")
                    print(f"Sample values: {str(known_encoding)[:60]}")
                    break

    def get_face_encoding(self, face_img: np.ndarray, face_info: Dict) -> Optional[np.ndarray]:
        """
        Extract face encoding from face image

        Args:
            face_img: Face image in BGR format
            face_info: Face information dictionary

        Returns:
            Face encoding as numpy array
        """
        try:
            embedding = None
            if isinstance(face_info, dict):
                embedding = face_info.get('embedding')
            else:
                embedding = getattr(face_info, 'embedding', None)

            if embedding is None:
                return None

            return np.asarray(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None

    def match_face(self, face_encoding: np.ndarray) -> Tuple[Optional[int], str, float]:
        print(f"Matching face encoding: shape={getattr(face_encoding, 'shape', None)}")
        print(f"Sample values: {str(face_encoding)[:60]}")
        """
        Match a face encoding against known faces

        Args:
            face_encoding: Face encoding to match

        Returns:
            Tuple of (profile_id, name, confidence)
        """
        if not self.known_face_encodings:
            return None, "Unknown", 0.0

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

                    if distance < best_distance and distance < self.threshold:
                        best_distance = distance
                        best_match_id = self.known_face_ids[i]
                        best_match_name = self.known_face_names[i]

        # Convert distance to confidence (inverse relationship)
        confidence = 1.0 - (best_distance / self.threshold) if best_distance < self.threshold else 0.0

        return best_match_id, best_match_name, confidence

    def add_new_face(self, face_img: np.ndarray, face_info: Dict,
                     name: str, angle: float = 0.0) -> Optional[int]:
        """
        Add a new face to the database_db

        Args:
            face_img: Face image in BGR format
            face_info: Face information dictionary
            name: Name of the person
            angle: Face angle in degrees

        Returns:
            Profile ID if successful, None otherwise
        """
        # Get face encoding
        face_encoding = self.get_face_encoding(face_img, face_info)
        if face_encoding is None:
            return None

        # Add profile to database_db
        profile_id = self.db_manager.add_profile(name=name)

        # Add face encoding
        self.db_manager.add_face_encoding(profile_id, face_encoding, angle)

        # Update in-memory data
        self.known_face_ids.append(profile_id)
        self.known_face_names.append(name)
        self.known_face_encodings.append([(face_encoding, angle)])

        return profile_id

    def update_face_angles(self, profile_id: int, face_img: np.ndarray,
                           face_info: Dict, angle: float):
        """
        Update a face in the database_db with a new angle

        Args:
            profile_id: ID of the profile to update
            face_img: Face image in BGR format
            face_info: Face information dictionary
            angle: Face angle in degrees
        """
        # Get face encoding
        face_encoding = self.get_face_encoding(face_img, face_info)
        if face_encoding is None:
            return False

        # Check if this angle is already recorded
        encodings = self.db_manager.get_profile_encodings(profile_id)
        existing_angles = [a for _, a in encodings]

        # Add new angle if significantly different
        if not any(abs(angle - a) < 15 for a in existing_angles):
            self.db_manager.add_face_encoding(profile_id, face_encoding, angle)

            # Update in-memory data
            idx = self.known_face_ids.index(profile_id)
            self.known_face_encodings[idx].append((face_encoding, angle))

            return True

        return False


if __name__ == "__main__":
    import os
    import sys
    try:
        import tkinter as tk
        from tkinter import filedialog
        HAS_TKINTER = True
    except ImportError:
        HAS_TKINTER = False

    # Use the same database as face_detector.py
    db_path = os.path.join(os.path.dirname(__file__), "../database_db/face_profiles.db")
    db = DatabaseManager(db_path)
    recognizer = FaceRecognizer(db)

    # Import InsightFace for embedding extraction
    from insightface.app import FaceAnalysis

    # Select image file
    def select_image_file():
        if HAS_TKINTER:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select an image for face recognition",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"), ("All files", "*.*")],
                initialdir=os.getcwd()
            )
            root.destroy()
            return file_path if file_path else None
        else:
            print("tkinter not available. Please enter image path:")
            return input("Image path: ").strip()

    image_path = select_image_file()
    if not image_path or not os.path.exists(image_path):
        print("No image selected or file does not exist.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    # Detect faces and extract embeddings using InsightFace
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)
    if not faces:
        print("No faces detected in the image.")
        sys.exit(0)

    print(f"Found {len(faces)} face(s) in {os.path.basename(image_path)}.")

    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
        embedding = face.embedding
        print(f"\n--- Face {i+1} ---")
        print(f"Location: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
        print(f"Embedding vector length: {len(embedding)}")
        match_id, name, confidence = recognizer.match_face(embedding)
        print(f"Match result: ID={match_id}, Name={name}, Confidence={confidence:.2f}")