NPY_DIR=$1
RAW_IMAGE_DIR=$2
SAVE_DIR=$3
PROP_MODEL_NAME=$4

REF_MODEL_NAME=cps-transunetpe-dlv3pb3_ce_lvsz_flare22slicesv2-ps2

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