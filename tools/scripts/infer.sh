
GT_DIR=$1
ROOT_DIR=$2
SAVE_DIR=runs/test_infer

MODEL_NAME=cps-transunetpe-dlv3pb3_ce_lvsz_flare22slicesv2-ps2
PROPAGATION_NAME=stcn_r50_ohmce_lvsz_aug-flare22slicesv2-ps2

PRED_DIR="${SAVE_DIR}/${MODEL_NAME}/test/masks"
OUTDIR="${SAVE_DIR}/${MODEL_NAME}/test/submission"
WEIGHT="weights/${MODEL_NAME}/checkpoints/best.pth"
VIS_DIR="${SAVE_DIR}/${MODEL_NAME}/test/mp4"

PROP_WEIGHT="weights/${PROPAGATION_NAME}/checkpoints/best.pth"

# python tools/inference/infer_1stage.py \
#               -c configs/semantic2D/flare22v2/normal/test.yaml \
#               -o global.weights=$WEIGHT \
#               data.dataset.args.root_dir=$ROOT_DIR \
#               global.save_dir=$SAVE_DIR \
#               global.exp_name=$MODEL_NAME \
#               global.exist_ok=True

# PYTHONPATH=. python tools/inference/infer_cps.py \
#               -c configs/semantic2D/flare22v2/cps/test.yaml \
#               -o global.weights=$WEIGHT \
#               data.dataset.args.root_dir=$ROOT_DIR \
#               global.save_dir=$SAVE_DIR \
#               global.exp_name=$MODEL_NAME \
#               global.exist_ok=True

PYTHONPATH=. python tools/inference/infer_2stage_efficient.py \
              -c configs/semantic2D/flare22v2/stcn/test.yaml \
              -o global.ref_weights=$WEIGHT \
              data.dataset.args.root_dir=$ROOT_DIR \
              global.save_dir=$SAVE_DIR \
              global.exp_name=$MODEL_NAME \
              global.prop_weights=$PROP_WEIGHT \
              global.exist_ok=True

# PYTHONPATH=. python tools/inference/infer_refine.py \
#               -c configs/semantic2D/flare22v2/stcn/test_refine.yaml \
#               -o data.dataset.args.root_dir=$ROOT_DIR \
#                 data.dataset.args.mask_dir=runs/unlabelled_infer/full/cps-transunetpe-dlv3pb3_ce_lvsz_flare22slicesv2-ps2/test/masks \
#                 global.save_dir=$SAVE_DIR \
#                 global.exp_name=$MODEL_NAME \
#                 global.prop_weights=$WEIGHT \
#                 global.exist_ok=True

# python tools/inference/infer_slices.py \
#               -c configs/cps/test_slices.yaml \
#               -o global.cfg_transform=configs/cps/transform_slices.yaml \
#               global.weights=$WEIGHT \
#               data.dataset.args.root_dir=$ROOT_DIR \
#               global.save_dir=$SAVE_DIR \
#               global.exp_name=$MODEL_NAME \
#               global.exist_ok=True

PYTHONPATH=. python tools/postprocess/make_submission.py \
                -p $PRED_DIR \
                -g $GT_DIR \
                -o $OUTDIR

# PYTHONPATH=. python tools/evaluation/DSC_NSD_eval.py \
#                 -p $OUTDIR \
#                 -g ../data/raw/ValMask 

# PYTHONPATH=. python tools/eda/visualize.py \
#                 -i $GT_DIR \
#                 -l $OUTDIR \
#                 -o $VIS_DIR