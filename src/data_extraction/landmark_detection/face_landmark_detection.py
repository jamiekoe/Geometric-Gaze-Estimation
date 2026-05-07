"""Face landmark detection with InsightFace."""

import insightface


class FaceLandmarkDetector:
    """Detect 68 face landmarks needed for head pose and eye cropping."""

    def __init__(self):
        """Load the InsightFace model used for face landmark detection."""
        self.model = insightface.app.FaceAnalysis(name="antelopev2", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 480))

    def detect(self, img):
        """Return 2D image coordinates for one frame's 68 face landmarks."""
        faces = self.model.get(img)
        if not faces:
            raise ValueError("No face detected in frame.")
        face = faces[0]
        landmarks = face["landmark_3d_68"][:, :2]  # only want (x,y) pixel coordinates
        return landmarks
