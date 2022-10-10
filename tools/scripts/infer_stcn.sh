NPY_DIR="/home/pmkhoi/github/ivos/data/flare22/npy/training/ValImage"
RAW_IMAGE_DIR="/home/pmkhoi/github/ivos/data/flare22/raw/training/images"
RAW_LABEL_DIR="/home/pmkhoi/github/ivos/data/flare22/raw/training/labels"
SAVE_DIR=runs/val_infer

REF_MODEL_NAME=cps-transunetpe-dlv3pb3_ce_lvsz_flare22slicesv2-ps2
PROP_MODEL_NAME=stcn_r50_ohmce_lvsz_aug-flare22slicesv2-ps2

REF_WEIGHT="weights/${REF_MODEL_NAME}/checkpoints/best.pth"
PROP_WEIGHT="weights/${PROP_MODEL_NAME}/checkpoints/best.pth"
PRED_DIR="${SAVE_DIR}/${PROP_MODEL_NAME}/test/masks"
OUTDIR="${SAVE_DIR}/${PROP_MODEL_NAME}/test/submission"
VIS_DIR="${SAVE_DIR}/${PROP_MODEL_NAME}/test/mp4"

echo "Inferencing..."
PYTHONPATH=. python tools/inference/infer_2stage_efficient.py \
              -c configs/flare22v2/stcn/test.yaml \
              -o global.ref_weights=$REF_WEIGHT \
              data.dataset.args.root_dir=$NPY_DIR \
              global.save_dir=$SAVE_DIR \
              global.exp_name=$PROP_MODEL_NAME \
              global.prop_weights=$PROP_WEIGHT \
              global.exist_ok=True

echo "Making submission"
PYTHONPATH=. python tools/postprocess/make_submission.py \
                -p $PRED_DIR \
                -g $RAW_IMAGE_DIR \
                -o $OUTDIR

echo "Evaluating..."
PYTHONPATH=. python tools/evaluation/DSC_NSD_eval.py \
                -p $OUTDIR \
                -g $RAW_LABEL_DIR

echo "Visualizing..."
PYTHONPATH=. python tools/eda/visualize.py \
                -i $RAW_IMAGE_DIR \
                -l $OUTDIR \
                -o $VIS_DIR