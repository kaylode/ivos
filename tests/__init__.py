import os
PROJECT_NAME = "flare2022"
TEST_ROOT = os.path.realpath(os.path.dirname(__file__))
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
DATA_RAW_DIR = os.path.join(PACKAGE_ROOT, "data/sample")
DATA_PRE_DIR = os.path.join(PACKAGE_ROOT, "data/sample_binary")
CONFIG_DIR = os.path.join(PACKAGE_ROOT, "configs", "semantic2D", "stcn")
SEED = 1234


# seed_everything(SEED)
