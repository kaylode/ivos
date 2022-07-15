python tools/train/stcn/train.py \
      -c configs/semantic2D/flare22v2/normal/pipeline_transunet.yaml \
      -o global.debug=True \
      trainer.args.print_interval=100 \
      trainer.args.evaluate_interval=1 \
      trainer.args.num_iterations=80000 \
      global.save_dir=./runs \
      global.exp_name=transunet32_pe_ohmce_lvsz-flare22slicesv2-ps2 \
      global.exist_ok=True \
      trainer.args.use_fp16=True \
      # global.resume=runs/transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt/checkpoints/last.pth