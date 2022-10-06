python tools/train/cps/train.py \
      -c configs/semantic2D/flare22v2/cps/pipeline.yaml \
      -o global.debug=True \
      trainer.args.print_interval=100 \
      trainer.args.evaluate_interval=1 \
      trainer.args.num_iterations=50000 \
      global.save_dir=./runs \
      global.exp_name=cps-transunetpe-dlv3pb3_ce_lvsz_flare22slicesv2-ps2 \
      global.exist_ok=True \
      trainer.args.use_fp16=True \
      global.pretrained1=weights/transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/best.pth \
      global.pretrained2=weights/dlv3pb3_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/best.pth \