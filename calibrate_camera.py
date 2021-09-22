import argparse
import os

import tinycam.calibration as clb
from tinycam.storage import JsonCameraStorage


strategies = {
    clb.CameraLens.RECTILINEAR: clb.RectLensStrategy,
    clb.CameraLens.FISHEYE: clb.FisheyeLensStrategy
}

transformers = {
    clb.CameraLens.RECTILINEAR: clb.RectTransformer,
    clb.CameraLens.FISHEYE: clb.FisheyeTransformer
}

def main(args):
    images = clb.load_folder_images(args.path)
    images = clb.resize_images(images)
    print(f"Found {len(images)} images for calibration.")

    print(f"Calculating camera...")
    calib_strategy = strategies[args.lens]()
    calib = clb.Calibration2d(*args.board_dim, calib_strategy)
    cam, cam_poses = calib.calculate_camera(images)

    print("Calibrated camera:")
    print(cam)
    

    if args.output:
        output_path = os.path.abspath(args.output)
        print(f"Saving camera to: {output_path}")
        jstore = JsonCameraStorage()
        jstore.dump(cam, output_path)
    print("Done: ")

if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument("-p", "--path", type=str, required=True,
        help="path to the checkerboard pattern images folder.")
    ag.add_argument("-l", "--lens", type=clb.CameraLens, default=clb.CameraLens.RECTILINEAR,
        help="Lens type rectilinear or fisheye.")
    ag.add_argument("-b", "--board-dim", nargs='+', type=int, default=(6, 8),
        help="Size of the checkerboard pattern used")
    ag.add_argument("-o", "--output", type=str, default=None, 
        help="Path to the ouptut json file with camera matrix")
    args = ag.parse_args()

    main(args)