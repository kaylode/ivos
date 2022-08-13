# python tools/preprocess/make_flare22.py \
#     -i ../data/raw/ \
#     -o ../data/flare2022-validation \
#     --ratio 0.9

PYTHONPATH=. python tools/preprocess/make_flare22_test.py \
    -i $1 \
    -o $2