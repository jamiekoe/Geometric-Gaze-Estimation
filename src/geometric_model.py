"""Geometric gaze model used by the ClearDepth experiments.

The model fits one eye at a time. It estimates the eyeball center in the head
coordinate system plus two angular offsets, then predicts where the resulting
visual axis intersects the display plane.
"""

import numpy as np
import scipy.optimize as opt
import os
import json
import copy


class GeometricModel:
    """Single-eye geometric model for screen point-of-gaze prediction."""

    @staticmethod
    def to_vector(E, alpha, beta):
        """Pack model parameters into the optimizer vector."""
        return np.append(E, np.array([alpha, beta]))

    @staticmethod
    def get_target_transitions(target_locations):
        """Find frame indices where the calibration target changes."""
        return np.append(0, np.argwhere(np.any(np.diff(target_locations, axis=0) != 0, axis=1)) + 1)

    @staticmethod
    def group(target_transitions, array):
        """Group per-frame values by calibration target interval."""
        output = []
        for i, start in enumerate(target_transitions):
            end = target_transitions[i + 1] if i + 1 < len(target_transitions) else len(array)
            output.append(array[start: end])
        return output

    def __init__(self, side, initial_E, model_path):
        """Create a left- or right-eye model with an initial eyeball center."""
        self.side = side
        self.initial_E = initial_E
        self.initial_alpha = 0
        self.initial_beta = 0
        self.params = None
        self.model_path = model_path

    def load_params(self):
        """Load fitted parameters for this eye from `model_path`."""
        with open(os.path.join(self.model_path, f"params_{self.side}.json"), "r") as f:
            params = json.load(f)
        missing = {"E", "alpha", "beta"} - set(params)
        if missing:
            raise ValueError(f"Missing model parameter keys for {self.side} eye: {sorted(missing)}.")
        params["E"] = np.array(params["E"])
        self.params = params

    def save_params(self):
        """Save fitted parameters for this eye to `model_path`."""
        with open(os.path.join(self.model_path, f"params_{self.side}.json"), "w") as f:
            params = copy.deepcopy(self.params)
            params["E"] = params["E"].tolist()
            f.write(json.dumps(params, indent=2))
            f.write("\n")

    def set_params(self, params):
        """Set model parameters from a dictionary with E, alpha, and beta."""
        params = copy.deepcopy(params)
        params["E"] = np.array(params["E"])
        self.params = params

    def compute_pog(self, x, dataset):
        """Predict 3D screen point-of-gaze for each frame in `dataset`.

        `x[:3]` is the eyeball center in the reference head coordinate system.
        The per-frame head pose maps it into the camera coordinate system before
        forming the optical axis from eyeball center to pupil center.
        """
        H = dataset["head_pose"]
        P = dataset["pupil"][self.side]
        E = H["t"] + H["R"] @ x[:3]  # N x 3
        O = P - E
        O /= np.linalg.norm(O, axis=1)[:, np.newaxis]  # N x 3
        denominator = np.sqrt(1 - O[:, 1] ** 2)  # N
        denominator[denominator == 0] = 1e-6
        theta = np.arccos(O[:, 0] / denominator)  # N
        phi = np.arcsin(O[:, 1])  # N
        r = np.cos(phi + x[4])  # N
        V = np.stack([r * np.cos(theta + x[3]), np.sin(phi + x[4]), -r * np.sin(theta + x[3])], axis=1)  # N x 3
        lam = -P[:, 2] / V[:, 2]  # N
        POG = P + lam[:, None] * V  # N x 3
        return POG

    def compute_loss(self, x, *args):
        """Optimizer objective: mean grouped Euclidean POG error in millimeters."""
        dataset = args[0]
        pog_pred = self.compute_pog(x, dataset)
        pog_gt = dataset["pog"]
        loss = np.linalg.norm(pog_pred - pog_gt, axis=1)
        loss = self.group(args[1], loss)
        return np.mean(loss)

    def compute_training_loss(self, dataset):
        """Evaluate the current parameters on the calibration dataset."""
        params = self.params
        x = self.to_vector(params["E"], params["alpha"], params["beta"])
        target_transitions = self.get_target_transitions(dataset["pog"])
        return float(self.compute_loss(x, dataset, target_transitions))

    def train(self, dataset, initial_params=None, maxiter=100000):
        """Fit eyeball center and angular offsets to calibration targets."""
        print("Training geometric model...")
        if initial_params is None:
            x0 = self.to_vector(self.initial_E, self.initial_alpha, self.initial_beta)
        else:
            x0 = self.to_vector(initial_params["E"], initial_params["alpha"], initial_params["beta"])
        options = {"maxiter": maxiter, "disp": True}
        target_transitions = self.get_target_transitions(dataset["pog"])
        args = (dataset, target_transitions)
        result = opt.minimize(self.compute_loss, x0, method="COBYLA", args=args, options=options)
        x = result.x
        self.params = {"E": x[:3], "alpha": x[3], "beta": x[4]}
        print("Training completed.")
        print("Success?", result.success)
        print("Message:", result.message)
        print("Model parameters:")
        print("E", np.round(x[:3], 2), "alpha", np.round(np.rad2deg(x[3])), "beta", np.round(np.rad2deg(x[4])))
        return result

    def predict(self, dataset):
        """Predict point-of-gaze using the fitted model parameters."""
        params = self.params
        x = self.to_vector(params["E"], params["alpha"], params["beta"])
        return self.compute_pog(x, dataset)
