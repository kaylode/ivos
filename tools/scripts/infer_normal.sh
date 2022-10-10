NPY_DIR="/home/pmkhoi/github/ivos/data/flare22/npy/training/ValImage"
RAW_IMAGE_DIR="/home/pmkhoi/github/ivos/data/flare22/raw/training/images"
RAW_LABEL_DIR="/home/pmkhoi/github/ivos/data/flare22/raw/training/labels"
SAVE_DIR=runs/val_infer

MODEL_NAME=transunetpe_ohmce_lvsz-flare22slicesv2-ps2-pt

WEIGHT="weights/${MODEL_NAME}/checkpoints/best.pth"
PRED_DIR="${SAVE_DIR}/${MODEL_NAME}/test/masks"
OUTDIR="${SAVE_DIR}/${MODEL_NAME}/test/submission"
VIS_DIR="${SAVE_DIR}/${MODEL_NAME}/test/mp4"

echo "Inferencing..."
PYTHONPATH=. python tools/inference/infer_1stage.py \
                -c configs/flare22v2/normal/test.yaml \
                -o global.weights=$WEIGHT \
                data.dataset.args.root_dir=$NPY_DIR \
                global.save_dir=$SAVE_DIR \
                global.exp_name=$MODEL_NAME \
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