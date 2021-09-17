# fisheye callibration
# from https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

import os
import argparse

import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import imutils
from imutils.paths import list_images
import json

ag = argparse.ArgumentParser()
ag.add_argument("-f","--folder",type=str, required=True,
    help="Path to the input folder with images")
ag.add_argument("-s",'--size', nargs='+', type=int, default=(6,8),
    help="Size of the checkerboard pattern used")
ag.add_argument("-o", "--output", type=str, default="./callib.json", 
    help="Path to the ouptu json file with camera matrix")
args = vars(ag.parse_args())

# constants
nrow, ncol = args["size"]
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# generate object points
# world frame is fixed to the checkerboard pattern corner
# later when each camera pose is known those can be transformed to a view where camera is in (0, 0)
objp = np.zeros((1, nrow*ncol, 3), np.float32)
objp[0,:,:2] = np.mgrid[0:nrow, 0:ncol].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

MAX_IMG_WIDTH = 1600
for path in list_images(args["folder"]):
    img = cv2.imread(path)
    if img.shape[0] > MAX_IMG_WIDTH:
        img = imutils.resize(img, width=MAX_IMG_WIDTH)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nrow, ncol),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))

rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(gray.shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
print("saving to json...")

with open(args["output"], 'w') as f:
    json.dump({"DIM":tuple(gray.shape[::-1]), "K":K.tolist(), "D": D.tolist()}, f)