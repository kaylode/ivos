# python tests/parser.py -c configs/template.yml -o global.debug=False
from theseus.opt import Opts

import pytest
import yaml
from tests import PACKAGE_ROOT
import os.path as osp 

@pytest.fixture
def minimal_cfg(tmp_path):
    with open(osp.join(PACKAGE_ROOT,'configs','cps','pipeline.yaml')) as f:
        cfg = yaml.safe_load(f)
    return cfg


def _save_cfg(cfg_path, cfg):
    with open(cfg_path, "w+") as f:
        yaml.safe_dump(cfg, f)


@pytest.mark.parametrize("exp_name", [None, "delete", "another_name"])
def test_opts_device_cpu(tmp_path, minimal_cfg, exp_name):
    def _fake_test():
        opts = Opts().parse_args(["-c", cfg_path,"-o", "global.exp_name=another_name"])
        assert opts["global"]["exp_name"] == "another_name"

    def _normal_test():
        opts = Opts().parse_args(["-c", cfg_path])
        assert opts["global"]["exp_name"] == "flare22-stcn"

    cfg_path = str(tmp_path / "default.yaml")
    _save_cfg(cfg_path, minimal_cfg)
    if exp_name == "another_name":
        _normal_test()  # opts.global.exp_name is default
        _fake_test()  # opts.global.exp_name is set to another_name
    elif exp_name is None:
        minimal_cfg["global"]["exp_name"] = None  # global has the "name" key but don't set
    elif exp_name == "delete":
        del minimal_cfg["global"]["exp_name"]  # global doesnt have the "name" key