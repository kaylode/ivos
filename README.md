# <p align="center"> MiVOS: Medical Interactive ctVolume Organ Segmentation </p>

# :pencil: Instructions

### **Installation**
- Go to root folder, run `pip install -e .`


### **Data Preparation**

- Analyze dataset
```
python theseus/semantic3D/utilities/eda.py \
    -i <image dir> \
    -l <mask dir>
```


- Preprocess AbdomenCT1k and split train/val
```
python theseus/semantic3D/utilities/preprocess/scripts/make_abdomen.py \
    -i data/abdomenct1k/Subtask1 \
    -o data/abdomenct1k/Subtask1-processed512clip \
    --ratio 0.95
```



### **Training**

#### Reference model
- Edit `normal/pipeline.yaml` then run
```
python configs/semantic2D/train.py \
      -c configs/semantic2D/normal/pipeline.yaml \
```

#### Propagation model
- Edit `stcn/pipeline.yaml` then run
```
python configs/semantic2D/train.py \
      -c configs/semantic2D/stcn/pipeline.yaml \
```

### **Inference**

- You can download the checkpoints from wandb by using
```
from theseus.utilities.download import download_from_wandb
download_from_wandb("checkpoints/best.pth", "kaylode/flare22/2k9dwq8w", "weights/stcn")
download_from_wandb("checkpoints/best.pth", "kaylode/flare22/2mkfc2ne", "weights/normal")
```

- To perform 2-stage inference, first modify `stcn/test.yaml`, then run
```
python configs/semantic2D/infer_2stage.py \
      -c configs/semantic2D/stcn/test.yaml \
      -o global.ref_weights=weights/normal/checkpoints/best.pth \
      global.prop_weights=weights/stcn/checkpoints/best.pth \
```

- After that, run below command to prepare submission
```
python theseus/semantic3D/utilities/postprocess/make_submission.py \
  -g <ground truth> \
  -p <prediction dir> \
  -o <output dir>
```

### **Evaluation**
- Standalone evaluation scripts officially provided by the organizers 
```
python tools/evaluation/DSC_NSD_eval.py \
  -g <ground truth mask> \
  -p <prediction mask>
```

### **References**

- https://github.com/kaylode/theseus
- https://github.com/hkchengrex/STCN
- https://github.com/hkchengrex/MiVOS

```
@inproceedings{cheng2021stcn,
  title={Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{cheng2021mivos,
  title={Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2021}
}
```