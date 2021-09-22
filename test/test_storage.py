import unittest

import pycamutils.calibration as clb
from pycamutils.storage import JsonCameraStorage
import numpy as np

import os

def assert_cameras_equal(tc: unittest.TestCase(), cam1: clb.Camera, cam2: clb.Camera):
    tc.assertEqual(cam1.dimensions, cam2.dimensions)
    tc.assertEqual(cam1.lens, cam2.lens)
    tc.assertTrue(isinstance(cam1.lens, clb.CameraLens))
    tc.assertTrue(isinstance(cam1.distortion, np.ndarray))
    tc.assertTrue(isinstance(cam1.camera_matrix, np.ndarray))
    tc.assertTrue(isinstance(cam2.lens, clb.CameraLens))
    tc.assertTrue(isinstance(cam2.distortion, np.ndarray))
    tc.assertTrue(isinstance(cam2.camera_matrix, np.ndarray))
    np.testing.assert_equal(cam1.distortion, cam2.distortion)
    np.testing.assert_equal(cam1.camera_matrix, cam2.camera_matrix)

class TestCameraStorage(unittest.TestCase):
    test_callib_load_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "test_load_callib.json"
    )
    test_callib_save_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "test_save_callib.json"
    )
    test_camera_matrix = np.array([
        [349.8306632364971, 0.0, 402.83220846029036],
        [0.0, 349.7236533950722, 301.59563333353515],
        [0.0, 0.0, 1.0]
    ])
    test_distortion = np.array([[
        0.06253452932933365,
        0.009416480849029437,
        -0.0057591063547560144,
        0.0017953020624106798
    ]]).T
    test_dim = (800, 600)

    def test_json_storage_load(self):
        cam1 = clb.Camera(
            self.test_camera_matrix,
            self.test_distortion,
            self.test_dim,
            clb.CameraLens.RECTILINEAR,
        )

        jstore = JsonCameraStorage()
        cam2 = jstore.load(self.test_callib_load_path)
        assert_cameras_equal(self, cam1, cam2)


    def test_json_storage_dump_then_load(self):
        cam1 = clb.Camera(
            self.test_camera_matrix,
            self.test_distortion,
            self.test_dim,
            clb.CameraLens.RECTILINEAR,
        )
        jstore = JsonCameraStorage()
        jstore.dump(cam1, self.test_callib_save_path)
        cam2 = jstore.load(self.test_callib_save_path)
        
        assert_cameras_equal(self, cam1, cam2)