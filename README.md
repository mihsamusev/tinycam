# This repo is for studying basics of structure from motion

Local install for running tests
```sh
pip isntall -e .
```

## Sources:


## Preliminaries
### Camera callibration

Calculation of camera matrix and distortion coefficients
https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html
https://www.theeminentcodfish.com/gopro-calibration/
Prerequisite theory []()


```python
python callibration.py -i <folder_with_checkerboard_images> -o <path_to_output.json or xml> -v <whether_to_visualize_or_not>
```

requirements
Output 3x3 camera matrix
1x5 distortion vector
supports fisheye
optionally rvectors and tvectors
optionally a visualization all checkerboard poses with camera position at (0, 0)

can use it as a library to undistort both normal and fisheye

LINK TO FAST API!
Upload images, get camera parameters, reprojection error, visualization report back

## Feature detection and description
```python
python feature_descriptor.py -i <path_to_image>
```

```python
python feature_matching.py -i <first_viewpoint_image> <second_viewpoint_image> -o <output_path>
```

Extract feature descriptors and those in 2 images, show valid matches with green and bad with red

## Epipolar geometry and 2 image 3d reconstruction 