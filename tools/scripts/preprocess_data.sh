TRAIN_PATH="data/flare22/raw/training"
VALIDATION_PATH="data/flare22/raw/validation/images"
GT_VALIDATION_PATH="data/flare22/raw/validation/labels"
UNLABELED_PATH="data/flare22/raw/unlabeled/images"

PROCESSED_TRAIN_PATH="data/flare22/processed/training/"
PROCESSED_VALIDATION_PATH="data/flare22/processed/validation/"
PROCESSED_UNLABELED_PATH="data/flare22/processed/unlabeled/images"

PROCESSED_IMAGE_TRAIN_PATH="data/flare22/processed/training/TrainImage"
PROCESSED_IMAGE_VAL_PATH="data/flare22/processed/training/ValImage"
PROCESSED_IMAGE_VALIDATION_PATH="data/flare22/processed/validation/ValidationImage"
PROCESSED_MASK_TRAIN_PATH="data/flare22/processed/training/TrainMask"
PROCESSED_MASK_VAL_PATH="data/flare22/processed/training/ValMask"
PROCESSED_MASK_VALIDATION_PATH="data/flare22/processed/validation/ValidationMask"

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
echo "Normalize training and validation"
PYTHONPATH=. python tools/preprocess/split_train_val.py \
    -i $TRAIN_PATH \
    -o $PROCESSED_TRAIN_PATH \
    --ratio 0.9

#Process validation data
echo "Normalize official validation"
PYTHONPATH=. python tools/preprocess/process_test.py \
    -i $VALIDATION_PATH \
    -l $GT_VALIDATION_PATH \
    -o $PROCESSED_VALIDATION_PATH \
    -t "Validation"

echo "Windowing CT labelled data"
python tools/preprocess/windowing_ct/run.py $PROCESSED_IMAGE_TRAIN_PATH $SLICES_IMAGE_TRAIN_PATH
python tools/preprocess/windowing_ct/run.py $PROCESSED_IMAGE_VAL_PATH $SLICES_IMAGE_VAL_PATH
python tools/preprocess/windowing_ct/run.py $PROCESSED_IMAGE_VALIDATION_PATH $SLICES_IMAGE_VALIDATION_PATH

mv $PROCESSED_MASK_TRAIN_PATH $SLICES_MASK_TRAIN_PATH
mv $PROCESSED_MASK_VAL_PATH $SLICES_MASK_VAL_PATH
mv $PROCESSED_MASK_VALIDATION_PATH $SLICES_MASK_VALIDATION_PATH


echo "Numpify labelled data"
python tools/preprocess/windowing_ct/numpify.py \
    -i $SLICES_IMAGE_TRAIN_PATH \
    -g $SLICES_MASK_TRAIN_PATH \
    -s $NPY_IMAGE_TRAIN_PATH \
    -sg $NPY_MASK_TRAIN_PATH

python tools/preprocess/windowing_ct/numpify.py \
    -i $SLICES_IMAGE_VAL_PATH \
    -g $SLICES_MASK_VAL_PATH \
    -s $NPY_IMAGE_VAL_PATH \
    -sg $NPY_MASK_VAL_PATH

python tools/preprocess/windowing_ct/numpify.py \
    -i $SLICES_IMAGE_VALIDATION_PATH \
    -g $SLICES_MASK_VALIDATION_PATH \
    -s $NPY_IMAGE_VALIDATION_PATH \
    -sg $NPY_MASK_VALIDATION_PATH

echo "Make csv file"
python tools/preprocess/windowing_ct/make_csv.py


#Process unlabelled data
UNLABELLED_PATH="data/flare22/raw/unlabelled/images"
mkdir data/flare22/processed/unlabelled
mkdir data/flare22/npy/unlabelled

PROCESSED_UNLABELLED_PATH="data/flare22/processed/unlabelled"
PROCESSED_UNLABELLED_IMAGE_PATH="data/flare22/processed/unlabelled/UnlabelledImage"
SLICES_UNLABELLED_IMAGE_PATH="data/flare22/slices/unlabelled/UnlabelledImage"
SLICES_UNLABELLED_NPY_PATH="data/flare22/npy/unlabelled/UnlabelledImage"

echo "Normalize unlabelled"
PYTHONPATH=. python tools/preprocess/process_test.py \
    -i $UNLABELLED_PATH \
    -o $PROCESSED_UNLABELLED_PATH \
    -t "Unlabelled"

echo "Windowing CT unlabelled"
python tools/preprocess/windowing_ct/run.py $PROCESSED_UNLABELLED_IMAGE_PATH $SLICES_UNLABELLED_IMAGE_PATH

python tools/preprocess/windowing_ct/numpify.py \
    -i $SLICES_UNLABELLED_IMAGE_PATH \
    -s $SLICES_UNLABELLED_NPY_PATH 