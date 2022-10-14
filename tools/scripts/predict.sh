#!/bin/bash

INPUT_PATH=/home/hcmus/workspace/inputs/
OUTPUT_PATH=/home/hcmus/workspace/outputs/
# INPUT_PATH=$1
# OUTPUT_PATH=$2

EXP_NAME=stcn_r50_ohmce_lvsz_aug-flare22slicesv2-ps2
PROCESSED_TEST_PATH=preproc/
PROCESSED_IMAGE_TEST_PATH=preproc/TestImage/
SLICES_IMAGE_TEST_PATH=preproc/TestImage_slices/
NPY_IMAGE_TEST_PATH=preproc/TestImage_npy/

echo "Normalizing..."
PYTHONPATH=. python tools/preprocess/process_test.py \
    -i $INPUT_PATH \
    -o $PROCESSED_TEST_PATH \
    -t "Test"

echo "Windowing CT..."
PYTHONPATH=. python tools/preprocess/windowing_ct/run.py \
    $PROCESSED_IMAGE_TEST_PATH \
    $SLICES_IMAGE_TEST_PATH

echo "Numpifying..."
python tools/preprocess/windowing_ct/numpify.py \
    -i $SLICES_IMAGE_TEST_PATH \
    -s $NPY_IMAGE_TEST_PATH

echo "Inferencing ..."
mkdir -p $OUTPUT_PATH
mkdir -p ./tmp_output/$EXP_NAME
sh ./tools/scripts/infer.sh $NPY_IMAGE_TEST_PATH $INPUT_PATH tmp_output/ $EXP_NAME

echo "Copying to output folder ..."
sudo cp -r ./tmp_output/$EXP_NAME/test/submission/* $OUTPUT_PATH

echo "Done!"