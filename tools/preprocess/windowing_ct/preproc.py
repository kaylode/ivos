from typing import List
import cv2
import numpy as np
import glob
import os
from pathlib import Path
from joblib import Parallel, delayed
import argparse
import nibabel as nib


def read_nib(img_path):
    r = nib.load(img_path.as_posix())
    # convert to hounsfield unit
    img = (np.asanyarray(r.dataobj) * r.dataobj.slope) + r.dataobj.inter
    return img


def read_dicom(img_path):
    import pydicom
    r = pydicom.read_file(img_path.as_posix())
    # convert to hounsfield unit
    img = (r.pixel_array * r.RescaleSlope) + r.RescaleIntercept
    return img


def read_npy(img_path):
    img = np.load(img_path.as_posix())
    return img


def preprocess(img_path, output_dir, WINDOW_LEVEL=None, WINDOW_WIDTH=None):
    image_name = img_path.stem.split(".")[0]
    # params
    window_min = None 
    window_max = None
    if isinstance(WINDOW_LEVEL, List) and isinstance(WINDOW_WIDTH, List):
        for i, (l, w) in enumerate(zip(WINDOW_LEVEL, WINDOW_WIDTH)):
            window_min = l - (w // 2) if window_min is None else min(window_min, l - (w // 2))
            window_max = l + (w // 2) if window_max is None else max(window_max, l + (w // 2))
    elif isinstance(WINDOW_LEVEL, int) and isinstance(WINDOW_WIDTH, int):
        window_min = WINDOW_LEVEL - (WINDOW_WIDTH // 2)
        window_max = WINDOW_LEVEL + (WINDOW_WIDTH // 2)
    else: 
        raise ValueError("WINDOW_LEVEL and WINDOW_WIDTH must be int or list of int")
    # read dicom file
    if img_path.suffix == ".dcm":
        imgs = read_dicom(img_path)
    elif img_path.suffix == ".gz":
        imgs = read_nib(img_path)
    elif img_path.suffix == ".npy":
        img = read_npy(img_path)
    else:
        raise Exception("Unknown file type")
    # apply window
    if img_path.suffix == ".gz":
        for i in range(imgs.shape[-1]):
            img = imgs[:, :, i]
            img = np.clip(img, window_min, window_max)
            img = 255 * ((img - window_min) / (window_max - window_min))
            img = img.astype(np.uint8)
            # write to output_dir
            cv2.imwrite((output_dir / f"{image_name}_{i:04d}.jpg").as_posix(), img)
    else:
        img = np.clip(img, window_min, window_max)
        img = 255 * ((img - window_min) / (window_max - window_min))
        img = img.astype(np.uint8)
        # write to output_dir
        cv2.imwrite((output_dir / f"{image_name}.jpg").as_posix(), img)
    return


def catch_wrapper(img_path, output_dir, WINDOW_LEVEL, WINDOW_WIDTH):
    try:
        preprocess(img_path, output_dir, WINDOW_LEVEL, WINDOW_WIDTH)
    except Exception as e:
        print(e, img_path.stem)


def run(args):
    WINDOW_LEVEL = args.window_level
    WINDOW_WIDTH = args.window_width

    input_dir = Path(args.data_dir)
    # output_dir = input_dir.parent.parent/f'preprocess/{input_dir.stem}_L{WINDOW_LEVEL}_W{WINDOW_WIDTH}'
    output_dir = (
        Path(args.output_dir) / input_dir.stem / args.name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    img_paths = list(input_dir.glob("*.dcm"))
    img_paths += list(input_dir.glob("*.nii.gz"))
    img_paths += list(input_dir.glob("*.npy"))
    x = "="
    n = 10
    print(
        f"{x*n} Start process {input_dir.stem} for level: {WINDOW_LEVEL}, width: {WINDOW_WIDTH} {x*n}"
    )
    # for f in img_paths:
    #     preprocess(f, output_dir, WINDOW_LEVEL, WINDOW_WIDTH)
    Parallel(n_jobs=8, verbose=1)(
        delayed(catch_wrapper)(f, output_dir, WINDOW_LEVEL, WINDOW_WIDTH)
        for f in img_paths
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--data-dir", "-d", default="data/origin/", type=str, help="Base data dir"
    )
    parser.add_argument(
        "--window-level", "-l", default=40, type=int, help="Window Level"
    )
    parser.add_argument(
        "--window-width", "-w", default=80, type=int, help="Window Width"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/preprocess/",
        type=str,
        help="Preprocess output dir",
    )

    args = parser.parse_args()
