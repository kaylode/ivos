python tools/train/stcn/train.py \
      -c weights/dlv3pb2/pipeline.yaml \
      -o global.debug=True \
      global.save_dir=./runs \
      global.exp_name=dlv3pb2-flare22slice \
      global.exist_ok=True \
      trainer.args.use_fp16=True \
      global.resume=weights/dlv3pb2/checkpoints/last.pth