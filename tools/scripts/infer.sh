
GT_DIR=$1
ROOT_DIR=$2
# OUTDIR=$3
SAVE_DIR=runs/test_infer

MODEL_NAME=unetppb2-flare22slices

PRED_DIR="runs/test_infer/${MODEL_NAME}/test/masks"
WEIGHT="runs/${MODEL_NAME}/checkpoints/best.pth"

python tools/inference/infer_1stage.py \
              -c configs/semantic2D/flare22/normal/test.yaml \
              -o global.weights=$WEIGHT \
              data.dataset.args.root_dir=$ROOT_DIR \
              global.save_dir=$SAVE_DIR \
              global.exp_name=$MODEL_NAME \
              global.exist_ok=True

# PYTHONPATH=. python tools/inference/infer_cps.py \
#               -c configs/cps/test.yaml \
#               -o global.weights=$WEIGHT \
#               data.dataset.args.root_dir=$ROOT_DIR \
#               global.save_dir=$SAVE_DIR \
#               global.exp_name=cps_flare22slices_full \
#               global.exist_ok=True

# python tools/inference/infer_2stage.py \
#               -c configs/semantic2D/flare22/stcn/test.yaml \
#               -o global.ref_weights=runs/cps_flare22slices_full/checkpoints/best.pth \
#               data.dataset.args.root_dir=$ROOT_DIR \
#               global.prop_weights=$WEIGHT \
#               global.save_dir=$SAVE_DIR \
#               global.exp_name=$MODEL_NAME \
#               global.exist_ok=True

# python tools/inference/infer_slices.py \
#               -c configs/cps/test_slices.yaml \
#               -o global.weights=runs/cps_flare22slices_full/checkpoints/best.pth \
#               data.dataset.args.root_dir=../data/slices/unlabelled/unlabelled_part2 \
#               global.save_dir=runs/pseudo_infer \
#               global.exp_name=cps_flare22slices_full \
#               global.exist_ok=True

# PYTHONPATH=. python tools/postprocess/make_submission.py \
#                 -p $PRED_DIR \
#                 -g $GT_DIR \
#                 -o $OUTDIR