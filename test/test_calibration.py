import unittest
import os

import numpy as np
import tinycam.calibration as clb

def assert_cameras_equal(tc: unittest.TestCase(), cam1: clb.Camera, cam2: clb.Camera):
    tc.assertEqual(cam1.dimensions, cam2.dimensions)
    tc.assertEqual(cam1.lens, cam2.lens)
    tc.assertTrue(isinstance(cam1.lens, clb.CameraLens))
    tc.assertTrue(isinstance(cam1.distortion, np.ndarray))
    tc.assertTrue(isinstance(cam1.camera_matrix, np.ndarray))
    tc.assertTrue(isinstance(cam2.lens, clb.CameraLens))
    tc.assertTrue(isinstance(cam2.distortion, np.ndarray))
    tc.assertTrue(isinstance(cam2.camera_matrix, np.ndarray))
    np.testing.assert_almost_equal(cam1.distortion, cam2.distortion, decimal=5)
    np.testing.assert_almost_equal(cam1.camera_matrix, cam2.camera_matrix, decimal=3)

class TestCalibration(unittest.TestCase):
    rectilinear_images = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "android_linear"
    )
    expected_rect_cam = clb.Camera(
        camera_matrix=np.array([
            [863.219, 0.0, 400.618],
            [0, 868.156, 563.184],
            [0, 0, 1]
        ]),
        distortion=np.array([[ 4.56631526e-01, -2.68414751e+00,  2.02551915e-02, -4.33383783e-04,  5.32189874e+00]]),
        dimensions=(800, 1066),
        lens=clb.CameraLens.RECTILINEAR
    )

    fisheye_images = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "gopro_wide"
    )
    expected_fisheye_cam = clb.Camera(
        camera_matrix=np.array([
            [349.977, 0, 403.511],
            [0, 349.642, 301.280],
            [0, 0, 1]
        ]),
        distortion=np.array([[ 0.06101743],[ 0.00975248],[ 0.00468699],[-0.00602115]]),
        dimensions=(800, 600),
        lens=clb.CameraLens.FISHEYE
    )

    resize_width = 800
    checkerboard_dim = (6, 8)
    #rect_camera = clb.Camera()
    #fisheye_camera = clb.Camera()

    def test_checkerboard_world_coords(self):
        expected = np.array([[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 2, 0],
            [1, 2, 0]
        ]])
        cb = clb.Calibration2d(2, 3, clb.RectLensStrategy())
        arr = cb.board_coords
        np.testing.assert_array_equal(arr.shape, expected.shape)
        np.testing.assert_array_equal(arr, expected)

    def test_rect_corners_output_shapes(self):

        images = clb.load_folder_images(self.rectilinear_images)
        images = clb.resize_images(images, max_width=800)
        
        cb = clb.Calibration2d(*self.checkerboard_dim, clb.RectLensStrategy())
        world_coords, image_coords, valid_images = cb.find_corners(images)

        n_img = len(valid_images)
        (h_img, w_img) = valid_images[0].shape[:2]
        self.assertEqual(len(world_coords), n_img)
        self.assertEqual(len(image_coords), n_img)
        self.assertTrue(image_coords[0][:, 0, 0].max() <= w_img)
        self.assertTrue(image_coords[0][:, 0, 1].max() <= h_img)

    def test_fisheye_corners_output_shapes(self):

        images = clb.load_folder_images(self.fisheye_images)
        images = clb.resize_images(images, max_width=800)
        
        cb = clb.Calibration2d(*self.checkerboard_dim, clb.FisheyeLensStrategy())
        world_coords, image_coords, valid_images = cb.find_corners(images)

        n_img = len(valid_images)
        (h_img, w_img) = valid_images[0].shape[:2]
        self.assertEqual(len(world_coords), n_img)
        self.assertEqual(len(image_coords), n_img)
        self.assertTrue(image_coords[0][:, 0, 0].max() <= w_img)
        self.assertTrue(image_coords[0][:, 0, 1].max() <= h_img)

    def test_rect_calibration(self):
        images = clb.load_folder_images(self.rectilinear_images)
        images = clb.resize_images(images, max_width=800)
        
        cb = clb.Calibration2d(*self.checkerboard_dim, clb.RectLensStrategy())
        cam, _ = cb.calculate_camera(images)
        assert_cameras_equal(self, cam, self.expected_rect_cam)

    def test_fisheye_calibration(self):
        images = clb.load_folder_images(self.fisheye_images)
        images = clb.resize_images(images, max_width=800)
        
        cb = clb.Calibration2d(*self.checkerboard_dim, clb.FisheyeLensStrategy())
        cam, _ = cb.calculate_camera(images)
        assert_cameras_equal(self, cam, self.expected_fisheye_cam)