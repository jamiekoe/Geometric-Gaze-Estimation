"""Combined face and eye landmark detection interface."""

import numpy as np
from .face_landmark_detection import FaceLandmarkDetector
from .eye_landmark_detection import EyeLandmarkDetector


class LandmarkDetector:
    """Thin wrapper that exposes batch face/eye landmark detection."""

    def __init__(self):
        """Create detector handles lazily so cached runs avoid model loading."""
        self.face_landmark_detector = None
        self.eye_landmark_detector = None

    def get_face_landmark_detector(self):
        """Return the InsightFace detector, loading it on first use."""
        if self.face_landmark_detector is None:
            self.face_landmark_detector = FaceLandmarkDetector()
        return self.face_landmark_detector

    def get_eye_landmark_detector(self):
        """Return the EyeNet detector, loading it on first use."""
        if self.eye_landmark_detector is None:
            self.eye_landmark_detector = EyeLandmarkDetector()
        return self.eye_landmark_detector

    def detect_face_landmarks(self, imgs):
        """Detect 68 face landmarks for a stack of BGR frames."""
        print("Detecting face landmarks...")
        detector = self.get_face_landmark_detector()
        return np.stack([detector.detect(img) for img in imgs], axis=0)

    def detect_eye_landmarks(self, imgs, face_landmarks):
        """Detect left/right eye landmarks using frame-aligned face landmarks."""
        print("Detecting eye landmarks...")
        subject_left, subject_right = self.get_eye_landmark_detector().detect(imgs, face_landmarks)
        return {"left": subject_left, "right": subject_right}
