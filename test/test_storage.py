import unittest
import pycamutils.calibration as clb
import numpy as np

import os

class TestCameraStorage(unittest.TestCase):
    test_callib_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data/test_callib.json"
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

    def test_json_storage(self):
        cam = clb.Camera(
            self.test_dim,
            clb.CameraLens.RECTILINEAR,
            clb.CameraIntrinsics(self.test_camera_matrix, self.test_distortion),
            clb.CameraExtrinsics()
        )

        jstore = clb.JsonCameraStorage()
        loaded_cam = jstore.load(self.test_callib_path)
        self.assertEqual(cam, loaded_cam)