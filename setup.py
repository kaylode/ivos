import setuptools

setuptools.setup(
    name="theseus",
    version='0.0.1',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        "imageio",
        "numpy",
        "torch",
        "tensorboard",
        "albumentations>=1.1.0",
        "torchvision",
        "tqdm",
        "timm==0.4.12",
        "pyyaml>=5.1",
        "webcolors",
        "omegaconf",
        "gdown>=4.4.0",
        "grad-cam",
        "tabulate",
        "segmentation-models-pytorch",
        "opencv-python-headless==4.2.0.32",
        "progressbar2",
        "opencv-python" ,
        "gitpython",
        "SimpleITK",
        "monai",
        "loguru",
        "wandb"
    ],
)