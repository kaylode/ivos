PRED_DIR=$1
OUT_DIR=$2
RAW_IMAGE_DIR=$3

echo "Ensembling..."
PYTHONPATH=. python tools/postprocess/ensemble_npy.py \
    -p $PRED_DIR \
    -o $OUT_DIR

OUT_MASK_DIR="${OUT_DIR}/masks"
SUBMISSION_DIR="${OUT_DIR}/submission"

echo "Making submission..."
PYTHONPATH=. python tools/postprocess/make_submission.py \
                -p $OUT_MASK_DIR \
                -g $RAW_IMAGE_DIR \
                -o $SUBMISSION_DIR

# To evaluate the ensemble result, uncomment these lines
# echo "Evaluating..."
# PYTHONPATH=. python tools/evaluation/DSC_NSD_eval.py \
#                 -p $SUBMISSION_DIR \
#                 -g data/flare22/raw/training/labels