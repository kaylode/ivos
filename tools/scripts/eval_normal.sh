PYTHONPATH=. python tools/train/stcn/eval.py \
      -c weights/transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt/pipeline.yaml \
      -o global.exp_name=$1 \
      global.save_dir=$2 \
      trainer.args.use_fp16=False \
      global.pretrained="weights/transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/best.pth" \
      global.cfg_transform="weights/transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt/transform.yaml"