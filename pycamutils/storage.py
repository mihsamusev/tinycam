import os
import json
from abc import ABC, abstractmethod

import numpy as np
from pycamutils.calibration import Camera, CameraLens

class CameraStorage(ABC):
    @abstractmethod
    def load(self, path: str) -> Camera:
        """Loads camera data from the path"""
    
    @abstractmethod
    def dump(self, cam: Camera, path: str) -> None:
        """Dumps camera data to the path"""

class JsonCameraStorage(CameraStorage):
    def load(self, path: str) -> Camera:
        if not os.path.exists(path):
            raise FileExistsError(f"{path} does not exists.")

        with open(path, 'r') as f:
            cam = json.load(f)
            camera_matrix = np.array(cam["camera_matrix"])
            distortion = np.array(cam["distortion"])
            dim = tuple(cam["dimensions"])
            lens = CameraLens(cam["lens"])
            return Camera(camera_matrix, distortion, dim, lens)

    def dump(self, cam: Camera, path: str, **kwargs) -> None:
        with open(path, 'w') as f:
            json.dump(
                {
                    "lens": cam.lens,
                    "dimensions": cam.dimensions,
                    "camera_matrix": cam.camera_matrix.tolist(),
                    "distortion": cam.distortion.tolist()
                }, f, **kwargs)

