import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer
import sqlite3


class FaceRecognitionSystem:
    def __init__(self):
        self.detector = MTCNN()
        self.face_model = load_model('models/facenet_keras.h5')
        self.normalizer = Normalizer(norm='l2')
        self.conn = sqlite3.connect('database_db/face_embeddings.db')

    def detect_faces(self, image):
        """Detect faces in an image"""
        faces = self.detector.detect_faces(image)
        return faces

    def extract_face(self, image, box, required_size=(160, 160)):
        """Extract and resize a face from an image"""
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height

        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, required_size)
        return face

    def get_embedding(self, face_pixels):
        """Get face embedding"""
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std

        samples = np.expand_dims(face_pixels, axis=0)
        embedding = self.face_model.predict(samples)
        return embedding[0]

    def recognize_face(self, embedding, threshold=0.7):
        """Recognize a face from embedding"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT face_id, embedding FROM faces")
        known_faces = cursor.fetchall()

        min_dist = float('inf')
        identity = None

        for face_id, known_embedding in known_faces:
            known_embedding = np.fromstring(known_embedding, sep=',')
            dist = np.linalg.norm(embedding - known_embedding)

            if dist < min_dist and dist < threshold:
                min_dist = dist
                identity = face_id

        return identity, min_dist

    def process_image(self, image):
        """Process an image to detect and recognize faces"""
        results = []
        faces = self.detect_faces(image)

        for i, face in enumerate(faces):
            box = face['box']
            confidence = face['confidence']

            if confidence < 0.9:
                continue

            face_img = self.extract_face(image, box)
            embedding = self.get_embedding(face_img)
            identity, distance = self.recognize_face(embedding)

            results.append({
                'box': box,
                'confidence': confidence,
                'embedding': embedding,
                'identity': identity,
                'distance': distance
            })

        return results

    def add_face_to_database(self, embedding, face_id, name=None):
        """Add a new face to the database_db"""
        embedding_str = ','.join(map(str, embedding))
        cursor = self.conn.cursor()

        cursor.execute("INSERT INTO faces (face_id, name, embedding) VALUES (?, ?, ?)",
                       (face_id, name, embedding_str))
        self.conn.commit()

    def close(self):
        """Close database_db connection"""
        self.conn.close()