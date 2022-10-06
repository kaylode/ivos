TRAIN_PATH="data/flare22/raw/training"
VALIDATION_PATH="data/flare22/raw/validation/images"
GT_VALIDATION_PATH="data/flare22/raw/validation/labels"
UNLABELED_PATH="data/flare22/raw/unlabeled/images"

PROCESSED_TRAIN_PATH="data/flare22/processed/training/"
PROCESSED_VALIDATION_PATH="data/flare22/processed/validation/"
PROCESSED_UNLABELED_PATH="data/flare22/processed/unlabeled/images"

PROCESSED_IMAGE_TRAIN_PATH="data/flare22/processed/training/TrainImage"
PROCESSED_IMAGE_VAL_PATH="data/flare22/processed/training/ValImage"
PROCESSED_MASK_TRAIN_PATH="data/flare22/processed/training/TrainMask"
PROCESSED_MASK_VAL_PATH="data/flare22/processed/training/ValMask"

SLICES_IMAGE_TRAIN_PATH="data/flare22/slices/training/TrainImage"
SLICES_IMAGE_VAL_PATH="data/flare22/slices/training/ValImage"
SLICES_IMAGE_VALIDATION_PATH="data/flare22/slices/validation/ValidationImage"
SLICES_MASK_TRAIN_PATH="data/flare22/slices/training/TrainMask"
SLICES_MASK_VAL_PATH="data/flare22/slices/training/ValMask"
SLICES_MASK_VALIDATION_PATH="data/flare22/slices/validation/ValidationMask"
SLICES_CSV_TRAIN="data/flare22/train_slices.csv"
SLICES_CSV_VAL="data/flare22/val_slices.csv"

NPY_IMAGE_TRAIN_PATH="data/flare22/npy/training/TrainImage"
NPY_IMAGE_VAL_PATH="data/flare22/npy/training/ValImage"
NPY_IMAGE_VALIDATION_PATH="data/flare22/npy/validation/ValidationImage"
NPY_MASK_TRAIN_PATH="data/flare22/npy/training/TrainMask"
NPY_MASK_VAL_PATH="data/flare22/npy/training/ValMask"
NPY_MASK_VALIDATION_PATH="data/flare22/npy/validation/ValidationMask"
NPY_CSV_TRAIN="data/flare22/train_npy.csv"
NPY_CSV_VAL="data/flare22/val_npy.csv"

# Process training data
echo "Split and numpify training and validation"
PYTHONPATH=. python tools/preprocess/split_train_val.py \
    -i $TRAIN_PATH \
    -o $PROCESSED_TRAIN_PATH \
    --ratio 0.95

#Process validation data
echo "Normalize official validation"
PYTHONPATH=. python tools/preprocess/process_test.py \
    -i $VALIDATION_PATH \
    -l $GT_VALIDATION_PATH \
    -o $PROCESSED_VALIDATION_PATH

# echo "Windowing CT labelled data"
# python tools/preprocess/windowing_ct/run.py $PROCESSED_TRAIN_IMAGE_PATH $SLICES_TRAIN_IMAGE_PATH
# python tools/preprocess/windowing_ct/run.py $PROCESSED_VAL_IMAGE_PATH $SLICES_VAL_IMAGE_PATH
# python tools/preprocess/windowing_ct/run.py $PROCESSED_VALIDATION_PATH $SLICES_VALIDATION_IMAGE_PATH

# mv $PROCESSED_TRAIN_MASK_PATH $SLICES_TRAIN_MASK_PATH
# mv $PROCESSED_VAL_MASK_PATH $SLICES_VAL_MASK_PATH
# mv $PROCESSED_VALIDATION_MASK_PATH $SLICES_VALIDATION_MASK_PATH


# echo "Numpify labelled data"
# python tools/preprocess/windowing_ct/numpify.py $SLICES_TRAIN_IMAGE_PATH $SLICES_TRAIN_NPY_PATH
# python tools/preprocess/windowing_ct/numpify.py $SLICES_VAL_IMAGE_PATH $SLICES_VAL_NPY_PATH
# python tools/preprocess/windowing_ct/numpify.py $SLICES_VALIDATION_IMAGE_PATH $SLICES_VALIDATION_NPY_PATH

# echo "Make csv file"
# python tools/preprocess/windowing_ct/make_csv.py \
#     -i $SLICES_TRAIN_IMAGE_PATH \
#     -g $PROCESSED_TRAIN_MASK_PATH \
#     -o $CSV_TRAIN

# python tools/preprocess/windowing_ct/make_csv.py \
#     -i $SLICES_VAL_IMAGE_PATH \
#     -g $PROCESSED_VAL_MASK_PATH \
#     -o $CSV_VAL



# #Process unlabelled data
# UNLABELLED_PATH="data/flare22/raw/unlabelled/images"
# PROCESSED_UNLABELLED_PATH="data/flare22/processed/unlabelled/images"
# SLICES_UNLABELLED_IMAGE_PATH="data/flare22/slices/unlabelled/UnlabelledImage"
# SLICES_UNLABELLED_NPY_PATH="data/flare22/npy/unlabelled/UnlabelledImage"

# echo "Normalize unlabelled"
# PYTHONPATH=. python tools/preprocess/process_test.py \
#     -i $UNLABELLED_PATH \
#     -o $PROCESSED_UNLABELLED_PATH

# echo "Windowing CT labelled data"
# python tools/preprocess/windowing_ct/run.py $PROCESSED_UNLABELLED_PATH $SLICES_UNLABELLED_IMAGE_PATH

# echo "Numpify unlabelled data"
# python tools/preprocess/windowing_ct/numpify.py $SLICES_UNLABELLED_IMAGE_PATH $SLICES_UNLABELLED_NPY_PATH