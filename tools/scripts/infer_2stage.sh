REF_WEIGHT=runs/cps_flare22slices_full/checkpoints/best.pth
PROP_WEIGHT=runs/stcn_r50-flare22slices-pseudo3/checkpoints/best.pth
ROOT_DIR=../data/nib_normalized/ValImage
SAVE_DIR=runs/val_infer
# PRED_DIR=runs/test_infer/cps_flare22slices_full/test2/masks
# GT_DIR=../data/raw/Validation
# OUTDIR=/home/nhtlong/flare2022/main/runs/stcn_r50-flare22slices-pseudo3/test

python tools/inference/infer_2stage_efficient.py \
              -c configs/semantic2D/flare22/stcn/test.yaml \
              -o global.ref_weights=$REF_WEIGHT \
              data.dataset.args.root_dir=$ROOT_DIR \
              global.prop_weights=$PROP_WEIGHT \
              global.save_dir=$SAVE_DIR \
              global.exp_name=stcn_r50 \
              global.exist_ok=True

# python tools/postprocess/make_submission.py \
#                 -p $PRED_DIR \
#                 -g $GT_DIR \
#                 -o $OUTDIR