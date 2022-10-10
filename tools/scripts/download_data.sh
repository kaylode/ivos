mkdir data
mkdir data/flare22
mkdir data/flare22/raw
mkdir data/flare22/raw/training
mkdir data/flare22/raw/training/images
mkdir data/flare22/raw/training/labels
mkdir data/flare22/raw/validation
mkdir data/flare22/raw/validation/images
mkdir data/flare22/raw/validation/labels

gdown 1J82QxWatc2WAMdgURhZ4P_F_dZbS7Ylz -O data/flare22/raw/training/images.zip
gdown 1hxAQH90KEx1cgccj_WtYsd3n9T_8xNb8 -O data/flare22/raw/training/labels.zip
gdown 1-EyyzXeH0vfpf4QH3jHJYHISlq43ai6u -O data/flare22/raw/validation/images.zip
gdown 1QMLaUVNIPtQOEy1fORh0dKVTW8DxvfGE -O data/flare22/raw/validation/images2.zip
gdown 1tyGII5If3ex8AvG-DFHg3tlfhSZylDes -O data/flare22/raw/validation/labels/labels.zip


unzip data/flare22/raw/training/images.zip -d data/flare22/raw/training/images
unzip data/flare22/raw/training/labels.zip -d data/flare22/raw/training/labels
unzip data/flare22/raw/validation/images.zip -d data/flare22/raw/validation/
mv data/flare22/raw/validation/Validation/* data/flare22/raw/validation/images
unzip data/flare22/raw/validation/images2.zip -d data/flare22/raw/validation/
mv data/flare22/raw/validation/Validation/* data/flare22/raw/validation/images
unzip data/flare22/raw/validation/labels/labels.zip -d data/flare22/raw/validation/labels

rm -r data/flare22/raw/validation/Validation
rm data/flare22/raw/training/images.zip
rm data/flare22/raw/training/labels.zip
rm data/flare22/raw/validation/images.zip
rm data/flare22/raw/validation/images2.zip
rm data/flare22/raw/validation/labels/labels.zip


# unlabelled data
mkdir data/flare22/raw/unlabelled
mkdir data/flare22/raw/unlabelled/images

gdown 1FKorcHfaDIqTWPsZDrlpr_DpMDFgrzb2 -O data/flare22/raw/unlabelled/images.zip
unzip data/flare22/raw/unlabelled/images.zip -d data/flare22/raw/unlabelled
mv data/flare22/raw/unlabelled/FLARE22_UnlabeledCase1751-2000/*  data/flare22/raw/unlabelled/images
rm data/flare22/raw/unlabelled/images.zip
rm -r data/flare22/raw/unlabelled/FLARE22_UnlabeledCase1751-2000

# gdown 1fL-x1eRK2713gPvfZi0eBgaUPH7Z0kJg -O data/flare22/raw/unlabelled/images.zip
# unzip data/flare22/raw/unlabelled/images.zip -d data/flare22/raw/unlabelled
# rm data/flare22/raw/unlabelled/images.zip


# https://drive.google.com/file/d/1kKgE0jM8LqAbKg0rJuuHj6oCDwDS0_ke/view?usp=sharing
# https://drive.google.com/file/d/1ytVtWfDW9BXK7yyhUyX6hHuZTJ9NmTtX/view?usp=sharing
# https://drive.google.com/file/d/1zLbN0nmy4m9mKwJz4CxTLfLTDaEDbNZx/view?usp=sharing
# https://drive.google.com/file/d/1zl2mtd2mnm-nRRTWzvt4Yja7io89c0mo/view?usp=sharing
# https://drive.google.com/file/d/1PdGvqtSO9y7KcxcALmR9r-53lTXlR-oq/view?usp=sharing
