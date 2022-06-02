python tools/train/cps/train.py \
      -c configs/cps/pipeline_slices.yaml \
      -o global.debug=True \
      trainer.args.print_interval=100 \
      trainer.args.evaluate_interval=1 \
      trainer.args.num_iterations=30000 \
      global.save_dir=./runs \
      global.exp_name=cps_flare22slices_full \
      global.exist_ok=True \
      trainer.args.use_fp16=True \
      global.resume=weights/cps/checkpoints/best.pth