import os 
import pytest #noqa
from tests.utils import run_cmd

@pytest.mark.skip(reason="Validation takes too long")
def test_train_semantic2D():
    # ignore this test for now
    run_cmd("python tools/train/train.py \
            -c configs/semantic2D/flare22/stcn/pipeline.yaml \
            -o global.debug=False \
            global.device=cpu \
            data.dataset.train.args.root_dir=data/sample_binary/ \
            data.dataset.train.args.csv_path=data/sample_binary/train.csv \
            data.dataset.val.args.root_dir=data/sample_binary/ \
            data.dataset.val.args.csv_path=data/sample_binary/val.csv \
            data.dataloader.train.args.batch_size=2 \
            trainer.args.print_interval=2 \
            trainer.args.save_interval=1 \
            trainer.args.num_iterations=1 \
            trainer.args.evaluate_interval=1 \
            global.save_dir=runs \
            global.exp_name=stcn-flare22_320clip_binary_pt \
            global.exist_ok=True \
            trainer.args.use_fp16=False", "test_train_semantic2D")