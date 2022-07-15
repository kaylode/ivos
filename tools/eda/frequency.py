import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser('Frequency')
parser.add_argument('-i', '--input', )


def class_frequency(args):
    filenames = os.listdir(args.input)
    class_dict = {}
    for filename in tqdm(filenames):
        filepath = osp.join(args.input, filename)


        casenames = os.listdir(filepath)
        for casename in casenames:
            casepath = osp.join(filepath, casename)
            np_file = np.load(casepath)
            labels = np.unique(np_file)

            for label in labels:
                if label not in class_dict.keys():
                    class_dict[label] = 0
                class_dict[label] += 1

    sorted_class_dict = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1])}
    print(sorted_class_dict)

if __name__ == '__main__':
    args = parser.parse_args()
    class_frequency(args)