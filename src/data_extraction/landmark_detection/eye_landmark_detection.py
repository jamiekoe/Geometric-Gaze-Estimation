"""Eye landmark detection wrapper around `david-wb/gaze-estimation`.

The external repository provides the EyeNet architecture and pretrained
checkpoint. This wrapper adapts its webcam-oriented preprocessing to the
RGB-D frames and 68-point face landmarks used by this project.
"""

from typing import List
import torch
import numpy as np
import cv2
import sys
from pathlib import Path

DEFAULT_EYE_LANDMARKS_REPO = Path(__file__).resolve().parents[3] / "third_party" / "gaze-estimation"
EyeNet = None
EyePrediction = None
EyeSample = None


# methods from https://github.com/david-wb/gaze-estimation/blob/249691893a37944a03e4ad4a3448083b6f63af10/run_with_webcam.py#L119
# (slightly modified)


def get_eye_landmarks_repo():
    """Return the bundled `david-wb/gaze-estimation` submodule path."""
    return DEFAULT_EYE_LANDMARKS_REPO


def load_eyenet_components():
    """Import EyeNet classes from the external gaze-estimation checkout."""
    global EyeNet, EyePrediction, EyeSample
    repo_path = get_eye_landmarks_repo()
    if EyeNet is not None:
        return repo_path, EyeNet, EyePrediction, EyeSample

    if not (repo_path / "models" / "eyenet.py").exists():
        raise FileNotFoundError(
            "Eye landmark dependency not found at "
            f"{repo_path}. From the repository root, run "
            "'git submodule update --init --recursive'."
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from models.eyenet import EyeNet as LoadedEyeNet
    from util.eye_prediction import EyePrediction as LoadedEyePrediction
    from util.eye_sample import EyeSample as LoadedEyeSample

    EyeNet = LoadedEyeNet
    EyePrediction = LoadedEyePrediction
    EyeSample = LoadedEyeSample
    return repo_path, EyeNet, EyePrediction, EyeSample


class EyeLandmarkDetector:
    """Detect eye-boundary landmarks and pupil/iris center for each frame."""

    def __init__(self):
        """Load the pretrained EyeNet checkpoint."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        repo_path, eye_net_cls, _, _ = load_eyenet_components()
        checkpoint_path = repo_path / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "EyeNet checkpoint not found at "
                f"{checkpoint_path}. Download it with "
                "'cd third_party/gaze-estimation && bash scripts/fetch_models.sh'."
            )
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        nstack = checkpoint["nstack"]
        nfeatures = checkpoint["nfeatures"]
        nlandmarks = checkpoint["nlandmarks"]
        self.eyenet = eye_net_cls(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(self.device)
        self.eyenet.load_state_dict(checkpoint["model_state_dict"])

    @staticmethod
    def segment_eyes(frame, landmarks, ow=160, oh=96):
        """Crop and normalize left/right eye images for EyeNet.

        The input face landmarks are 68-point image coordinates. The returned
        `EyeSample` objects carry inverse transforms so EyeNet landmarks can be
        mapped back into the original image coordinate system.
        """
        if EyeSample is None:
            load_eyenet_components()
        eyes = []
        # Segment eyes
        # for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
            x1, y1 = landmarks[corner1]
            x2, y2 = landmarks[corner2]
            eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # center image on middle of eye
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # Scale
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            estimated_radius = 0.5 * eye_width * scale

            # center image
            center_mat = np.asmatrix(np.eye(3))
            center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            inv_center_mat = np.asmatrix(np.eye(3))
            inv_center_mat[:2, 2] = -center_mat[:2, 2]

            # Get rotated and scaled, and segmented image
            transform_mat = center_mat * scale_mat * translate_mat
            inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

            eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
            eye_image = cv2.equalizeHist(eye_image)

            if is_left:
                eye_image = np.fliplr(eye_image)
            eyes.append(EyeSample(orig_img=frame.copy(),
                                  img=eye_image,
                                  transform_inv=inv_transform_mat,
                                  is_left=is_left,
                                  estimated_radius=estimated_radius))
        return eyes

    def run_eyenet(self, eyes: List["EyeSample"], ow=160, oh=96) -> List["EyePrediction"]:
        """Run EyeNet and map predicted landmarks back to image coordinates."""
        if EyePrediction is None:
            load_eyenet_components()
        result = []
        for eye in eyes:
            with torch.no_grad():
                x = torch.tensor(eye.img[None], dtype=torch.float32).to(self.device)
                _, landmarks, gaze = self.eyenet.forward(x)
                landmarks = np.asarray(landmarks.cpu().numpy()[0])
                gaze = np.asarray(gaze.cpu().numpy()[0])
                if gaze.shape != (2,) or landmarks.shape != (34, 2):
                    raise ValueError(
                        f"Unexpected EyeNet output shapes: gaze={gaze.shape}, landmarks={landmarks.shape}."
                    )

                landmarks = landmarks * np.array([oh / 48, ow / 80])

                temp = np.zeros((34, 3))
                if eye.is_left:
                    temp[:, 0] = ow - landmarks[:, 1]
                else:
                    temp[:, 0] = landmarks[:, 1]
                temp[:, 1] = landmarks[:, 0]
                temp[:, 2] = 1.0
                landmarks = temp
                landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
                if landmarks.shape != (34, 2):
                    raise ValueError(f"Unexpected transformed landmark shape: {landmarks.shape}.")
                result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
        return result

    def detect(self, imgs, face_landmarks):
        """Return detected subject-left and subject-right eye landmarks."""
        print("Detecting eye landmarks...")
        eye_landmarks_left = []
        eye_landmarks_right = []
        for img, lms in zip(imgs, face_landmarks):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eye_samples = self.segment_eyes(gray, lms)
            eye_preds = self.run_eyenet(eye_samples)
            left_eye = list(filter(lambda x_: x_.eye_sample.is_left, eye_preds))[0].landmarks[:-1] # removing eyeball center
            right_eye = list(filter(lambda x_: not x_.eye_sample.is_left, eye_preds))[0].landmarks[:-1] # removing eyeball center
            eye_landmarks_left.append(left_eye)
            eye_landmarks_right.append(right_eye)
        # The project uses subject-left/subject-right naming. EyeNet's
        # `is_left` flag follows image-left crop handling, so return in the
        # project convention expected by `LandmarkDetector`.
        return np.stack(eye_landmarks_right), np.stack(eye_landmarks_left)
