"""
Database management module for face recognition system
"""

import sqlite3
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class DatabaseManager:
    """Manages SQLite database_db for face profiles and detections"""

    def __init__(self, db_path: str = "face_profiles.db"):
        db_location = Path(db_path)
        db_location.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_location)
        self.init_db()

    def init_db(self):
        """Initialize the SQLite database_db"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Create tables
        c.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            created_at TIMESTAMP,
            last_seen TIMESTAMP,
            gender TEXT,
            age_range TEXT,
            ethnicity TEXT,
            notes TEXT
        )
        ''')

        c.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER,
            encoding BLOB,
            angle INTEGER,
            FOREIGN KEY (profile_id) REFERENCES profiles (id)
        )
        ''')

        c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER,
            timestamp TIMESTAMP,
            confidence REAL,
            angle INTEGER,
            image_path TEXT,
            FOREIGN KEY (profile_id) REFERENCES profiles (id)
        )
        ''')

        conn.commit()
        conn.close()

    def add_profile(self, name: str, gender: str = "Unknown",
                    age_range: str = "Unknown", ethnicity: str = "Unknown",
                    notes: str = "") -> int:
        """Add a new profile to the database_db"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
        INSERT INTO profiles (name, created_at, last_seen, gender, age_range, ethnicity, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            name,
            datetime.now(),
            datetime.now(),
            gender,
            str(age_range),
            ethnicity,
            notes
        ))

        profile_id = c.lastrowid
        conn.commit()
        conn.close()

        return profile_id

    def add_face_encoding(self, profile_id: int, encoding, angle: int = 0):
        """Add a face encoding to a profile"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        encoding_blob = pickle.dumps(encoding)
        c.execute('''
        INSERT INTO face_encodings (profile_id, encoding, angle)
        VALUES (?, ?, ?)
        ''', (profile_id, encoding_blob, angle))

        conn.commit()
        conn.close()

    def get_all_profiles(self) -> List[Dict]:
        """Get all profiles from the database_db"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM profiles ORDER BY created_at DESC")
        profiles = [dict(row) for row in c.fetchall()]

        conn.close()
        return profiles

    def get_profile_encodings(self, profile_id: int) -> List[Tuple]:
        """Get all face encodings for a profile"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
        SELECT encoding, angle FROM face_encodings 
        WHERE profile_id = ?
        ''', (profile_id,))

        results = c.fetchall()
        conn.close()

        return [(pickle.loads(encoding), angle) for encoding, angle in results]

    def update_profile_last_seen(self, profile_id: int):
        """Update the last seen timestamp for a profile"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
        UPDATE profiles
        SET last_seen = ?
        WHERE id = ?
        ''', (datetime.now(), profile_id))

        conn.commit()
        conn.close()

    def record_detection(self, profile_id: int, confidence: float,
                         angle: int, image_path: str):
        """Record a face detection in the database_db"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
        INSERT INTO detections (profile_id, timestamp, confidence, angle, image_path)
        VALUES (?, ?, ?, ?, ?)
        ''', (profile_id, datetime.now(), confidence, angle, image_path))

        conn.commit()
        conn.close()

        # Update last seen once insertion transaction is complete
        self.update_profile_last_seen(profile_id)

    def get_detections(self, limit: int = 100) -> List[Dict]:
        """Get recent detections from the database_db"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute('''
        SELECT d.*, p.name 
        FROM detections d
        JOIN profiles p ON d.profile_id = p.id
        ORDER BY d.timestamp DESC
        LIMIT ?
        ''', (limit,))

        detections = [dict(row) for row in c.fetchall()]

        conn.close()
        return detections

    def update_profile_notes(self, profile_id: int, notes: str):
        """Update notes for a profile"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
        UPDATE profiles
        SET notes = ?
        WHERE id = ?
        ''', (notes, profile_id))

        conn.commit()
        conn.close()

    def update_profile(self, profile_id: int, name: Optional[str] = None,
                       gender: Optional[str] = None, age_range: Optional[str] = None,
                       ethnicity: Optional[str] = None, notes: Optional[str] = None):
        """Update profile fields selectively"""
        fields = []
        values = []
        if name is not None:
            fields.append("name = ?")
            values.append(name)
        if gender is not None:
            fields.append("gender = ?")
            values.append(gender)
        if age_range is not None:
            fields.append("age_range = ?")
            values.append(str(age_range))
        if ethnicity is not None:
            fields.append("ethnicity = ?")
            values.append(ethnicity)
        if notes is not None:
            fields.append("notes = ?")
            values.append(notes)
        if not fields:
            return
        values.append(profile_id)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        query = f"UPDATE profiles SET {', '.join(fields)} WHERE id = ?"
        c.execute(query, tuple(values))
        conn.commit()
        conn.close()

    def reassign_profile_encodings(self, source_profile_id: int, target_profile_id: int):
        """Reassign all encodings from source to target"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        UPDATE face_encodings SET profile_id = ? WHERE profile_id = ?
        ''', (target_profile_id, source_profile_id))
        conn.commit()
        conn.close()

    def reassign_profile_detections(self, source_profile_id: int, target_profile_id: int):
        """Reassign all detections from source to target"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        UPDATE detections SET profile_id = ? WHERE profile_id = ?
        ''', (target_profile_id, source_profile_id))
        conn.commit()
        conn.close()

    def delete_profile(self, profile_id: int):
        """Delete a profile and its related encodings and detections"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Delete related data first
        c.execute('DELETE FROM face_encodings WHERE profile_id = ?', (profile_id,))
        c.execute('DELETE FROM detections WHERE profile_id = ?', (profile_id,))
        c.execute('DELETE FROM profiles WHERE id = ?', (profile_id,))
        conn.commit()
        conn.close()

    def get_profile_by_id(self, profile_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM profiles WHERE id = ?', (profile_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None


if __name__ == "__main__":
    # Test database_db functionality
    db = DatabaseManager(":memory:")  # Use in-memory database_db for testing

    # Add a test profile
    profile_id = db.add_profile(
        name="Test Person",
        gender="Male",
        age_range="25-35",
        ethnicity="White",
        notes="Test profile"
    )

    print(f"Added profile with ID: {profile_id}")

    # Get all profiles
    profiles = db.get_all_profiles()
    print(f"Total profiles: {len(profiles)}")

    # Add a face encoding
    import numpy as np

    test_encoding = np.random.rand(128)
    db.add_face_encoding(profile_id, test_encoding, angle=0)

    # Get encodings
    encodings = db.get_profile_encodings(profile_id)
    print(f"Retrieved {len(encodings)} encodings for profile {profile_id}")