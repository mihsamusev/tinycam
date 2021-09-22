from __future__ import annotations
import os
from typing import Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import cv2
import imutils
from imutils.paths import list_images


class CameraLens(str, Enum):
    RECTILINEAR = "RECTILINEAR"
    FISHEYE = "FISHEYE"

@dataclass 
class CameraPoses:
    """
    Information describing a camera pose
    """
    translations: list = field(default_factory=list)
    rotations: list = field(default_factory=list)

    def rotation_matrices(self):
        """
        Converts rotation vectors to matrices using cv2.Rodrigues representation
        """
        matrices = []
        for r in self.rotations:
            matrices.append(cv2.Rodrigues(r))
        return matrices

@dataclass
class Camera:
    camera_matrix: np.ndarray 
    distortion: np.ndarray
    dimensions: Tuple[int, int]
    lens: CameraLens

    def __str__(self):
        str = "camera_matrix:\n{}\ndistortion: {}\ndimensions: {}\nlens: {}".format(
            np.array_str(self.camera_matrix, precision=3, suppress_small=True),
            np.array_str(self.distortion, precision=5, suppress_small=True),
            self.dimensions,
            self.lens
        )
        return str

class Transformer(ABC):
    @abstractmethod
    def undistort(self, image: np.ndarray, cam: Camera) -> np.ndarray:
        """Undistorts camera points"""


class RectTransformer(Transformer):
    def undistort(self, image: np.ndarray, cam: Camera) -> np.ndarray:
        (h, w) = image.shape[:2]
        new_camera_matrix, _=cv2.getOptimalNewCameraMatrix(
            cam.camera_matrix,
            cam.distortion,
            cam.dimensions,
            1,
            (w, h)
        )
        xmap, ymap = cv2.initUndistortRectifyMap(
            cam.camera_matrix,
            cam.distortion,
            None,
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2
        )
        undistorted_img = cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return undistorted_img


class FisheyeTransformer(Transformer):
    def undistort(self, image: np.ndarray, cam: Camera) -> np.ndarray:
        (h, w) = image.shape[:2]
        new_camera_matrix, _=cv2.getOptimalNewCameraMatrix(
            cam.camera_matrix,
            cam.distortion,
            cam.dimensions,
            1,
            (w, h)
        )
        xmap, ymap = cv2.fisheye.initUndistortRectifyMap(
            cam.camera_matrix,
            cam.distortion,
            None,
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2
        )
        undistorted_img = cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return undistorted_img
    

class CalibrationStrategy(ABC):
    """Lens calculation strategy"""
    @abstractmethod
    def calibrate(self, world_coords, image_coords) -> Tuple[Camera, CameraPoses]:
        """Returns a calibrated camera object"""

class RectLensStrategy(CalibrationStrategy):
    calib_flags = 0
    def calibrate(self, world_coords, image_coords, image_dim) -> Tuple[Camera, CameraPoses]:
        """
        Returns rectilinear camera object
        """    
        rerror, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            world_coords,
            image_coords,
            image_dim,
            None,
            None,
            flags=self.calib_flags
        )

        cam = Camera(camera_matrix, distortion, image_dim, CameraLens.RECTILINEAR)
        cam_poses = CameraPoses(tvecs, rvecs)
        return (cam, cam_poses)
    
class FisheyeLensStrategy(CalibrationStrategy):
    calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    def calibrate(self, world_coords, image_coords, image_dim) -> Tuple[Camera, CameraPoses]:    
        rerror, camera_matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            world_coords,
            image_coords,
            image_dim,
            None,
            None,
            flags=self.calib_flags
        )
        
        cam = Camera(camera_matrix, distortion, image_dim, CameraLens.FISHEYE)
        cam_poses = CameraPoses(tvecs, rvecs)
        return (cam, cam_poses)


class Calibration2d:
    corner_search = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 
    corner_accuracy = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    search_window = (11,11)
    dead_zone = (-1, -1)

    def __init__(self, board_rows: int, board_cols: int, calibration_strategy: CalibrationStrategy):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.calibration_strategy = calibration_strategy
        self.board_coords = self.get_board_coords()
    
    def get_board_coords(self) -> np.ndarray:
        """
        The world frame is fixed to the checkerboard pattern plane at left upper corner:
        Outputs (1, 2 * 3, 3) np.ndarray. First dim is for stacking images later.
        [[
            [0. 0. 0.]
            [1. 0. 0.]
            [0. 1. 0.]
            [1. 1. 0.]
            [0. 2. 0.]
            [1. 2. 0.]
        ]]
        """
        points = np.zeros((1, self.board_rows * self.board_cols, 3), np.float32)
        # build x and y
        points[0,:,:2] = np.mgrid[0:self.board_rows, 0:self.board_cols].T.reshape(-1, 2)
        return points

    def find_corners(self, images: List[np.ndarray]):
        valid_images = []
        world_coords = [] # 3d point in real world space
        image_coords = [] # 2d points in image plane.

        # loop through all images, assume they are gray and resized already
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            success, corners = cv2.findChessboardCorners(
                gray, 
                (self.board_rows, self.board_cols),
                None
            )

            if success:
                valid_images.append(image)
                world_coords.append(self.board_coords)
                refined_corners = cv2.cornerSubPix(
                    gray, corners,
                    self.search_window,
                    self.dead_zone,
                    self.corner_accuracy)
                image_coords.append(refined_corners)

        if not valid_images:
            raise ValueError("Images dont contain valid corners!")   
        return world_coords, image_coords, valid_images

    def draw_corners(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Draws checkerboard corners on top of the images
        """
        _, image_coords, images = self.find_corners(images)
        new_images = []
        for corners, image in zip(image_coords, images):
            new_image = cv2.drawChessboardCorners(
                image.copy(),
                (self.board_rows, self.board_cols),
                corners, True)
            new_images.append(new_image)
        return new_images

    def calculate_camera(self, images: List[np.ndarray]) -> Tuple[Camera, CameraPoses]:
        if not images:
            raise ValueError("No images to calculate calibration!")

        world_coords, image_coords, _ = self.find_corners(images)
        image_dim = (images[-1].shape[1], images[-1].shape[0])
        return self.calibration_strategy.calibrate(
            world_coords, image_coords, image_dim)



def load_folder_images(folder):
    """
    Ideally can load both from video or from folder image
    """
    if not os.path.exists(folder):
        raise FileExistsError(f"Folder does not exist: {folder} !")

    images = []
    for path in list_images(folder):
        img = cv2.imread(path)
        images.append(img)
    return images

def resize_images(images: List[np.ndarray], max_width=800):
    resized = []
    for image in images:
        if image.shape[1] > max_width:
            image = imutils.resize(image, width=max_width)
            resized.append(image)
    return resized

def draw(images: np.ndarray):
    for image in images:
        cv2.imshow('corners',image)
        if cv2.waitKey(0) == ord('a'):
            continue

def save_images(path: str,images: List[np.ndarray], prefix: str = ""):
    if not os.path.exists:
        raise FileExistsError("Save path does not exist!")
    for i, image in enumerate(images):
        name = os.path.join(path, f"{prefix}_{i}.png")
        cv2.imwrite(name, image)

if __name__ == "__main__":
    pass