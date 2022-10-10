import os
import os.path as osp
import pandas as pd
from tqdm import tqdm

folders = [
    # "FLARE22_UnlabeledCase1-250Processed",
    # "FLARE22_UnlabeledCase251-500Processed",
    # "FLARE22_UnlabeledCase501-750Processed",
    # "FLARE22_UnlabeledCase751-1000Processed",
    # "FLARE22_UnlabeledCase1001-1500Processed",
    # "FLARE22_UnlabeledCase1501-1750Processed",
    # "FLARE22_UnlabeledCase1751-2000Processed",
    "ValidationProcessed"
]

idx_ranges = [
    # [1, 251],
    # [251, 501],
    # [501, 751],
    # [751, 1000],
    # [1001, 1501],
    # [1501, 1751],
    # [1751, 2001],
    [1, 51],
]

df = {
    'image1': [],
    'image2': [],
    'image3': [],
}

ROOT_DIR = "."

for folder, idx_range in tqdm(zip(folders, idx_ranges)):
    folder_path = osp.join(ROOT_DIR, folder)
    for idx in range(idx_range[0], idx_range[1]):

        # if osp.exists(osp.join('./PseudoUnlabeledMask', f'Case_{str(idx).zfill(5)}_0000')):
        #     print(idx)
        #     continue

        folder1 = f"FLARETs_{str(idx).zfill(4)}_0000_abdomen-soft tissues_abdomen-liver"
        folder2 = f"FLARETs_{str(idx).zfill(4)}_0000_chest-lungs_chest-mediastinum"
        folder3 = f"FLARETs_{str(idx).zfill(4)}_0000_spine-bone"
        
        tmp_path = osp.join(folder_path, folder1)

        sids = len(os.listdir(tmp_path))

        mid_range = [int(sids/5), int(5*sids/5)]

        for sid in range(mid_range[0], mid_range[1]):
            image_name = f"FLARETs_{str(idx).zfill(4)}_0000_{str(sid).zfill(4)}.jpg"
            image_path1 = osp.join(folder, folder1, image_name)
            image_path2 = osp.join(folder, folder2, image_name)
            image_path3 = osp.join(folder, folder3, image_name)

            df['image1'].append(image_path1)
            df['image2'].append(image_path2)
            df['image3'].append(image_path3)

pd.DataFrame(df).to_csv("validation.csv", index=False)