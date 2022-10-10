PYTHONPATH=. python tools/train/stcn/eval.py \
      -c weights/stcn_r50_ohmce_lvsz_aug-flare22slicesv2-ps2/pipeline.yaml \
      -o global.exp_name=$1 \
      global.save_dir=$2 \
      trainer.args.use_fp16=False \
      global.pretrained="weights/stcn_r50_ohmce_lvsz_aug-flare22slicesv2-ps2/checkpoints/best.pth" \
      global.cfg_transform="weights/stcn_r50_ohmce_lvsz_aug-flare22slicesv2-ps2/transform.yaml"