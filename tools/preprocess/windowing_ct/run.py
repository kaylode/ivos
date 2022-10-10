from preproc import run
from pathlib import Path
from config import get_cfg
from types import SimpleNamespace
import sys
if __name__ == "__main__":
    
    print('=' * 10)
    root_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(exist_ok=True, parents=True)
    queries = [
        ["spine-bone"],
        ["abdomen-soft tissues", "abdomen-liver"],
        ["chest-lungs", "chest-mediastinum"],
    ]
    for q in queries:
        c = get_cfg(q)
        for data_folder in root_dir.iterdir():
            folder_name = data_folder.stem
            args = {
                "name": '_'.join(q),
                "data_dir": data_folder,
                "output_dir": output_dir, 
                "window_level": [x["L"] for x in c],
                "window_width": [x["W"] for x in c],
            }

            args = SimpleNamespace(**args)
            run(args)
            # break
        # break
