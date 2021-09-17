import argparse
import cv2
import imutils
from imutils.paths import list_images
import numpy as np

if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("-f","--folder",type=str, required=True)
    ag.add_argument("-s",'--size', nargs='+', type=int, default=(6,8))
    args = vars(ag.parse_args())

    nrow, ncol = args["size"]

    # for corner accuracy increase 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nrow*ncol,3), np.float32)
    objp[:,:2] = np.mgrid[0:nrow,0:ncol].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    MAX_IMG_WIDTH = 1600
    for path in list_images(args["folder"]):
        img = cv2.imread(path)
        if img.shape[0] > MAX_IMG_WIDTH:
            img = imutils.resize(img, width=MAX_IMG_WIDTH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        success, corners = cv2.findChessboardCorners(gray, (nrow, ncol),None)

        # If found, add object points, image points (after refining them)
        if success:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nrow, ncol), corners2,success)

            cv2.imshow('img',img)
            if cv2.waitKey(0) == ord('a'):
                continue

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1],None,None)

    print("camera matrix: ", mtx)
    print("distortion coefficients: ", dist)
    # get camera matrix and ROI for given width / height
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    method = 2
    for path in list_images(args["folder"]):
        img = cv2.imread(path)
        img = imutils.resize(img, width=800)

        # undistort
        if method == 1:
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        else:
            mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

        # crop the image
        cv2.imshow('distorted',img)
        cv2.imshow('undistorted',dst)
        if cv2.waitKey(0) == ord('a'):
            continue