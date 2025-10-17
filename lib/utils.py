# lib/utils.py
"""
Utility functions for the face recognition system
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import tempfile
import os
from pathlib import Path
from typing import Optional

def create_output_directory(base_dir: str = "output") -> Path:
    """Create output directory if it doesn't exist"""
    output_dir = Path(base_dir)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_frame(frame: np.ndarray, output_dir: Path, frame_count: int) -> str:
    """Save a frame to disk"""
    frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
    cv2.imwrite(str(frame_path), frame)
    return str(frame_path)


def draw_face_boxes(image: np.ndarray, faces: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on faces"""
    result_image = image.copy()

    for face in faces:
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox.astype(int)

        # Draw bounding box
        color = (0, 255, 0) if face['name'] != 'Unknown' else (0, 0, 255)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{face['name']} ({face['confidence']:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Draw label background
        cv2.rectangle(
            result_image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return result_image


def augment_face_image(face_img: np.ndarray) -> List[np.ndarray]:
    """
    Generate augmented versions of a face image

    Args:
        face_img: Face image in BGR format

    Returns:
        List of augmented face images
    """
    augmented = []

    # Original image
    augmented.append(face_img)

    # Horizontal flip
    flipped = cv2.flip(face_img, 1)
    augmented.append(flipped)

    # Brightness adjustment
    bright = cv2.convertScaleAbs(face_img, alpha=1.2, beta=20)
    augmented.append(bright)

    # Dark adjustment
    dark = cv2.convertScaleAbs(face_img, alpha=0.8, beta=-20)
    augmented.append(dark)

    # Gaussian blur
    blurred = cv2.GaussianBlur(face_img, (5, 5), 0)
    augmented.append(blurred)

    return augmented


def create_detection_timeline(detections: List[Dict]) -> plt.Figure:
    """Create a timeline plot of detections"""
    if not detections:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No detection data available',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Convert to DataFrame
    df = pd.DataFrame(detections)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Group by name and date
    timeline_df = df.groupby(['date', 'name']).size().reset_index(name='count')

    # Create pivot table
    pivot_df = timeline_df.pivot(index='date', columns='name', values='count').fillna(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df.plot(kind='line', ax=ax, marker='o')

    plt.title('Detection Timeline')
    plt.xlabel('Date')
    plt.ylabel('Number of Detections')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_demographic_charts(profiles: List[Dict]) -> Dict[str, plt.Figure]:
    """Create demographic charts from profile data"""
    charts = {}

    if not profiles:
        return charts

    df = pd.DataFrame(profiles)

    # Gender distribution
    if 'gender' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        gender_counts = df['gender'].value_counts()
        gender_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        plt.title('Gender Distribution')
        plt.axis('off')
        charts['gender'] = fig

    # Ethnicity distribution
    if 'ethnicity' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ethnicity_counts = df['ethnicity'].value_counts()
        ethnicity_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        plt.title('Ethnicity Distribution')
        plt.axis('off')
        charts['ethnicity'] = fig

    # Age distribution
    if 'age_range' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Extract numeric age if possible
        df['age_numeric'] = df['age_range'].apply(
            lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else np.nan
        )
        age_data = df['age_numeric'].dropna()

        if not age_data.empty:
            age_data.hist(bins=10, ax=ax)
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Count')
            charts['age'] = fig

    return charts


def get_video_info(video_path: str) -> Dict:
    """Get information about a video file"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {}

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info


def list_available_videos(video_dir: str) -> List[Dict]:
    """List available videos in a directory"""
    video_dir = Path(video_dir)
    videos = []

    if video_dir.exists():
        for video_file in video_dir.glob("*.mp4"):
            info = get_video_info(str(video_file))
            info['path'] = str(video_file)
            info['name'] = video_file.name
            videos.append(info)

    return videos


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test video listing
    videos = list_available_videos("data/test_videos")
    print(f"Found {len(videos)} test videos")

    # Test face augmentation
    test_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    augmented = augment_face_image(test_face)
    print(f"Generated {len(augmented)} augmented face images")

    # Test demographic charts
    test_profiles = [
        {'name': 'Person 1', 'gender': 'Male', 'age_range': '25', 'ethnicity': 'White'},
        {'name': 'Person 2', 'gender': 'Female', 'age_range': '30', 'ethnicity': 'Asian'},
        {'name': 'Person 3', 'gender': 'Male', 'age_range': '35', 'ethnicity': 'Black'}
    ]

    charts = create_demographic_charts(test_profiles)
    print(f"Created {len(charts)} demographic charts")



# Import for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("tkinter not available. File selection will use command line input.")

# ... (keep all existing utility functions) ...

def select_image_file() -> Optional[str]:
    """
    Open a file dialog to select an image file.
    Falls back to command-line input if tkinter is not available.

    Returns:
        Path to selected file or None if cancelled.
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
        ("All files", "*.*")
    ]

    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image for face processing",
        filetypes=file_types,
        initialdir=os.getcwd()
    )

    # Destroy the hidden window
    root.destroy()

    return file_path if file_path else None


if __name__ == "__main__":
    # Test the file selector
    print("Testing file selection utility...")
    selected_file = select_image_file()
    if selected_file:
        print(f"Selected file: {selected_file}")
    else:
        print("No file selected.")