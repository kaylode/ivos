PYTHONPATH=. python tools/train/cps/train.py \
      -c configs/flare22v2/cps/pipeline.yaml \
      -o global.exp_name=$1 \
      global.save_dir=$2 \
      trainer.args.use_fp16=True \
      global.pretrained1="weights/transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/best.pth" \
      global.pretrained2="weights/dlv3pb3_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/best.pth" \
      global.cfg_transform="configs/flare22v2/cps/transform.yaml"