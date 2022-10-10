PYTHONPATH=. python tools/train/stcn/train.py \
      -c configs/flare22v2/stcn/pipeline.yaml \
      -o global.exp_name=$1 \
      global.save_dir=$2 \
      global.exist_ok=True \
      trainer.args.use_fp16=True