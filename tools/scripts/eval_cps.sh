PYTHONPATH=. python tools/train/cps/eval.py \
      -c weights/cps-transunetpe-dlv3pb3_ce_lvsz_flare22slicesv2-ps2/pipeline.yaml \
      -o global.exp_name=$1 \
      global.save_dir=$2 \
      trainer.args.use_fp16=False \
      global.pretrained1="weights/transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/best.pth" \
      global.pretrained2="weights/dlv3pb3_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/best.pth" \
      global.cfg_transform="weights/cps-transunetpe-dlv3pb3_ce_lvsz_flare22slicesv2-ps2/transform.yaml"