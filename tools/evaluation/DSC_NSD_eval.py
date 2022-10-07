import numpy as np
import nibabel as nb
import os
import os.path as osp
from collections import OrderedDict
from SurfaceDice import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient,
)
from tabulate import tabulate
import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser("Official evaluation code from FLARE")
parser.add_argument("-g", "--gt_dir", type=str, help="Ground Truth directory")
parser.add_argument("-p", "--pred_dir", type=str, help="Prediction directory")

NUM_CLASSES = 13

LOG = "runs/vallog.csv"
df = pd.read_csv(LOG)


def eval(args):
    seg_path = args.pred_dir
    gt_path = args.gt_dir

    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith(".nii.gz")]
    filenames.sort()

    seg_metrics = OrderedDict()
    seg_metrics["Name"] = list()
    seg_metrics["DSC_mean"] = list()
    seg_metrics["NSD-1mm_mean"] = list()
    for i in range(1, NUM_CLASSES + 1):
        seg_metrics["DSC_{}".format(i)] = list()
        seg_metrics["NSD-1mm_{}".format(i)] = list()

    for name in filenames:
        seg_metrics["Name"].append(name)
        seg_metrics["DSC_mean"].append(0)
        seg_metrics["NSD-1mm_mean"].append(0)
        # load grond truth and segmentation

        gt_name = name.replace("_0000.nii.gz", ".nii.gz")
        gt_nii = nb.load(osp.join(gt_path, gt_name))
        case_spacing = gt_nii.header.get_zooms()
        gt_data = np.uint8(gt_nii.get_fdata())
        seg_data = np.uint8(nb.load(osp.join(seg_path, name)).get_fdata())

        for i in range(1, NUM_CLASSES + 1):
            if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
                DSC_i = 1
                NSD_i = 1
            elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
                DSC_i = 0
                NSD_i = 0
            else:
                surface_distances = compute_surface_distances(
                    gt_data == i, seg_data == i, case_spacing
                )
                DSC_i = compute_dice_coefficient(gt_data == i, seg_data == i)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
            seg_metrics["DSC_{}".format(i)].append(DSC_i)
            seg_metrics["NSD-1mm_{}".format(i)].append(NSD_i)

            seg_metrics["DSC_mean"][-1] += DSC_i
            seg_metrics["NSD-1mm_mean"][-1] += NSD_i

        seg_metrics["DSC_mean"][-1] /= NUM_CLASSES
        seg_metrics["NSD-1mm_mean"][-1] /= NUM_CLASSES

    df_dict = {}

    seg_metrics["Name"].insert(0, "All")
    seg_metrics["DSC_mean"].insert(0, np.mean(seg_metrics["DSC_mean"]))
    seg_metrics["NSD-1mm_mean"].insert(0, np.mean(seg_metrics["NSD-1mm_mean"]))

    df_dict['DSC_mean'] = seg_metrics["DSC_mean"][0]
    df_dict['NSD-1mm_mean'] = seg_metrics["NSD-1mm_mean"][0]

    for i in range(1, NUM_CLASSES + 1):
        seg_metrics["DSC_{}".format(i)].insert(
            0, np.mean(seg_metrics["DSC_{}".format(i)])
        )
        seg_metrics["NSD-1mm_{}".format(i)].insert(
            0, np.mean(seg_metrics["NSD-1mm_{}".format(i)])
        )
        df_dict["DSC_{}".format(i)] = seg_metrics["DSC_{}".format(i)][0]
        df_dict["NSD-1mm_{}".format(i)] = seg_metrics["NSD-1mm_{}".format(i)][0]

    date=datetime.now()
    df2 = pd.DataFrame([[date] + list(df_dict.values())], columns=["Date"] + list(df_dict.keys()))
    pd.concat([df, df2]).to_csv(LOG, index=False)

    # Print table
    table = tabulate(seg_metrics, headers="keys", tablefmt="fancy_grid")

    print(table)


if __name__ == "__main__":
    args = parser.parse_args()
    eval(args)
