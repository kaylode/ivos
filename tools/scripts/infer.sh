
GT_DIR=$1
ROOT_DIR=$2
OUTDIR=$3
SAVE_DIR=runs/test_infer
PRED_DIR=runs/test_infer/cps_flare22slices_full/test/masks
WEIGHT=runs/cps_flare22slices_full/checkpoints/best.pth


# python tools/inference/infer_1stage.py \
#               -c configs/semantic2D/flare22/normal/test.yaml \
#               -o global.weights=runs/unetppb2-flare22slice/checkpoints/best.pth \
#               data.dataset.args.root_dir=../data/flare2022-validation/Validation \
#               global.save_dir=runs/test_infer \
#               global.exp_name=unetppb2_test \
#               global.exist_ok=True

PYTHONPATH=. python tools/inference/infer_cps.py \
              -c configs/cps/test.yaml \
              -o global.weights=$WEIGHT \
              data.dataset.args.root_dir=$ROOT_DIR \
              global.save_dir=$SAVE_DIR \
              global.exp_name=cps_flare22slices_full \
              global.exist_ok=True

# python tools/inference/infer_2stage.py \
#               -c configs/semantic2D/flare22/test.yaml \
#               -o global.ref_weights=weights/cps/checkpoints/best.pth \
#               global.prop_weights=weights/stcn/checkpoints/best.pth \
#               global.save_dir=../runs \
#               global.exp_name=2stage_infer_test \
#               global.exist_ok=True

# python tools/inference/infer_slices.py \
#               -c configs/cps/test_slices.yaml \
#               -o global.weights=runs/cps_flare22slices_full/checkpoints/best.pth \
#               data.dataset.args.root_dir=../data/slices/unlabelled/unlabelled_part2 \
#               global.save_dir=runs/pseudo_infer \
#               global.exp_name=cps_flare22slices_full \
#               global.exist_ok=True

PYTHONPATH=. python tools/postprocess/make_submission.py \
                -p $PRED_DIR \
                -g $GT_DIR \
                -o $OUTDIR