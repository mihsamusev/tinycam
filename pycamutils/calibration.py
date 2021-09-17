from typing import Tuple
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import json
import xml
import os

class CameraLens(str, Enum):
    RECTILINEAR = "RECTILINEAR"
    FISHEYE = "FISHEYE"


@dataclass    
class CameraExtrinsics:
    """
    Information describing a camera pose
    """
    translation: np.ndarray = np.zeros((1,4))
    rotation: np.ndarray = np.eye(3)


@dataclass
class CameraIntrinsics:
    """
    Information describing a camera intrinsic parameters
    """
    camera_matrix: np.ndarray 
    distortion: np.ndarray


@dataclass 
class Camera:
    shape: Tuple[int, int]
    lens: CameraLens
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


class CameraStorage:
    def load(self, path: str) -> Camera:
        pass
    def dump(self, cam: Camera, path: str) -> None:
        pass

class JsonCameraStorage(CameraStorage):

    def load(self, path: str) -> Camera:
        if not os.path.exists(path):
            raise FileExistsError(f"{path} does not exists.")

        with open(path, 'r') as f:
            cam = json.load(f)
            intrinsics = CameraIntrinsics(
                np.array(cam["camera_matrix"]),
                np.array(cam["distortion"]))
            extrinsics = CameraExtrinsics()
            return Camera(
                cam["shape"],
                cam["lens"],
                intrinsics,
                extrinsics,
                )

    def dump(self, cam: Camera, path: str, **kwargs) -> None:
        with open(path, 'w') as f:
            json.dump(
                {
                    "lens": cam.lens,
                    "shape": cam.shape,
                    "camera_matrix": cam.intrinsics.camera_matrix.tolist(),
                    "distortion": cam.intrinsics.distortion.tolist()
                }, f, **kwargs)

def main():
    k = np.array([
        [349.8306632364971, 0.0, 402.83220846029036],
        [0.0, 349.7236533950722, 301.59563333353515],
        [0.0, 0.0, 1.0]])
    d = np.array(
        [[0.06253452932933365], [0.009416480849029437], [-0.0057591063547560144], [0.0017953020624106798]])
    dim = (800, 600)

    c = Camera(
        dim,
        CameraLens.RECTILINEAR,
        CameraIntrinsics(k, d),
        CameraExtrinsics()
    )

    cs = JsonCameraStorage()
    cs.dump(c, "test_callib.json", indent=2)
    c_read = cs.load("test_callib.json")
    print(c_read)

if __name__ == "__main__":
    main()