from dis import get_instructions
import pytest
import torchvision
from theseus.base.datasets import DATALOADER_REGISTRY
from theseus.opt import Opts
from theseus.semantic2D.datasets import DATASET_REGISTRY
from theseus.semantic3D.augmentations import TRANSFORM_REGISTRY
from theseus.utilities.getter import get_instance_recursively
from torch.utils.data import DataLoader
from pathlib import Path


@pytest.mark.parametrize("dataset_name", ["FLARE22TrainDataset"])
def test_dataset(tmp_path, dataset_name):
    pipeline_cfg_path = "configs/semantic2D/flare22/stcn/pipeline.yaml"
    transform_cfg_path = "configs/semantic2D/flare22/transform.yaml"
    assert Path(pipeline_cfg_path).exists(), "config file not found"
    cfg_tf = Opts().parse_args(["-c",transform_cfg_path])
    cfg = Opts().parse_args(["-c",pipeline_cfg_path,"-o", 
            "global.debug=False",
            "global.device=cpu",
            "data.dataset.train.args.root_dir=data/sample_binary/",
            "data.dataset.train.args.csv_path=data/sample_binary/train.csv",
            "data.dataset.val.args.root_dir=data/sample_binary/",
            "data.dataset.val.args.csv_path=data/sample_binary/val.csv",
            "data.dataloader.train.args.batch_size=2"
    ])
    tf = get_instance_recursively(cfg_tf['train'], registry=TRANSFORM_REGISTRY)
    ds = get_instance_recursively(cfg['data']['dataset']['train'], registry=DATASET_REGISTRY, transform=tf)

    cfg['data']["dataloader"]['train'].update({'batch_size': 1})
    dataloader = get_instance_recursively(
        cfg['data']["dataloader"]['train'],
        registry=DATALOADER_REGISTRY,
        dataset=ds,
    )

    # data = {
    #         'inputs': images, # normalized image, torch.Tensor (T, C, H, W) 
    #         'targets': tar_masks, # target mask, numpy (T, 1, H, W) , values 1 at primary class
    #         'cls_gt': cls_gt, # numpy (T, H, W), each pixel present one class
    #         'sec_gt': sec_masks, # second object mask, numpy (T, 1, H, W) , values 1 at second class
    #         'selector': selector, # [1, 1] if has second object, else [1, 0]
    #         'info': {
    #             'name': patient_id,
    #             'slice_id': frames_idx,
    #             'affine': affine,
    #             'case_spacing': case_spacing
    #         },
    #     }
    for i, batch in enumerate(dataloader):
        print(batch['inputs'].shape)
        print(batch['targets'].shape)
        print(batch['info'])

