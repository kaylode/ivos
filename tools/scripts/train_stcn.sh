python tools/train/stcn/train.py \
      -c configs/semantic2D/flare22v2/stcn/pipeline.yaml \
      -o global.debug=True \
      trainer.args.print_interval=100 \
      trainer.args.save_interval=500 \
      trainer.args.num_iterations=60000 \
      trainer.args.evaluate_interval=1 \
      global.save_dir=./runs \
      global.exp_name=stcn_swin_ohmce_lvsz_aug-flare22slicesv2-ps2 \
      global.exist_ok=True \
      trainer.args.use_fp16=True \
      # global.resume=weights/stcn/checkpoints/best.pth