python tools/train/stcn/train.py \
      -c configs/semantic2D/flare22v2/normal/pipeline.yaml \
      -o global.debug=True \
      trainer.args.print_interval=100 \
      trainer.args.evaluate_interval=1 \
      trainer.args.num_iterations=30000 \
      global.save_dir=./runs \
      global.exp_name=dlv3pb2-flare22slicesv2-ps \
      global.exist_ok=True \
      trainer.args.use_fp16=True \
      # global.resume=weights/dlv3pb2/checkpoints/last.pth