import argparse
import json

import cv2
import imutils
from imutils.paths import list_images
import numpy as np

def undistort(img, K, D, DIM):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


if __name__ == '__main__':
    ag = argparse.ArgumentParser()
    ag.add_argument("-i","--input",type=str, required=True)
    ag.add_argument("-c","--callib", type=str, required=True)
    ag.add_argument("-o",'--output',type=str, required=True)
    args = vars(ag.parse_args())

    # You should replace these 3 lines with the output in calibration step
    with open(args["callib"], 'r') as f:
        callib = json.load(f)

    DIM = tuple(callib["DIM"])
    K = np.array(callib["K"])
    D = np.array(callib["D"])

    Knew, roi = cv2.getOptimalNewCameraMatrix(K, D, DIM, 0, DIM)

    for img_path in list_images(args["input"]):
        img = cv2.imread(img_path)
        img = imutils.resize(img, width=800)

        # recalculate camera matrix for resized image
        h, w = img.shape[:2]
        Knew, roi = cv2.getOptimalNewCameraMatrix(K, D, DIM, 0, (w, h))

        undistorted_img = undistort(img, Knew, D, (w, h))

        cv2.imshow("distorted", img)
        cv2.imshow("undistorted", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()