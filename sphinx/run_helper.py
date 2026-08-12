import os
import sys
import torch
import mosek
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

from mona.common.logging import get_logger


LOGGER = get_logger('run_helper')


def parse_args():
    parser = ArgumentParser(description='infer')
    parser.add_argument('config', type=str)
    parser.add_argument('-p', '--prev_data_csv_path', type=str, default=None)
    # parser.add_argument('--is_test', action='store_true')
    return parser.parse_args()


def get_mdl_num(cfg) -> int:
    return len(os.listdir(cfg["model"]["path"]))


def assert_mdl_conf(cfg, xhg, model_num) -> None:
    for h in cfg["model"]["horizon"]:
        assert f"{h}m" in cfg["model"]["path"]

    if xhg in ["okx10m", "binance5m", "okx5m", "CF5m"]:
        assert model_num == 8
    elif xhg == "CF":
        assert model_num == 4
    else:
        assert xhg is not None
        raise ValueError("Unknown exchange: " + xhg)


def add_model_legacy_path(patch_path_s: str) -> None:
    sys.path.append(patch_path_s)
    LOGGER.info(f"add {patch_path_s} to syspath")


def check_load_model(cfg):
    model_name = 'model'
    checkpoints = sorted(os.listdir(cfg[model_name]["path"]))
    for model_dir_p in checkpoints:
        mdl_path = os.path.join(cfg[model_name]["path"], model_dir_p, f"{cfg[model_name]['epoch']}.pth.tar")
        torch.load(mdl_path, map_location="cpu", weights_only=False)
        LOGGER.info(f"load checkpoint :{mdl_path}")


def check_load_mosek_license():
    os.environ["MOSEKLM_LICENSE_FILE"] = "mosek.lic"
    env = mosek.Env()
    # env.putlicensedebug(1)
    # env.checkoutlicense(mosek.feature.ptopt)
    version = env.getversion()
    LOGGER.info(f"mosek license loaded, version:{version}")


def sanity_check(cfg, xhg, model_num):
    assert_mdl_conf(cfg, xhg, model_num)
    check_load_model(cfg)
    check_load_mosek_license()


def dump_log(*args):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(f"error_{t}.txt", "w") as f:
        print(*args)
        print(*args, file=f)


def get_cn_rnd_up_min_ts():
    ts = pd.Timestamp.now(tz="Asia/Shanghai")
    task_min_ts = ts.floor("min") + pd.Timedelta(minutes=1)
    task_min_s = task_min_ts.strftime("%Y-%m-%d %H:%M:%S")
    return task_min_s


def gen_minute_encode_pred(model, norm_feature):
    norm_feature[~np.isfinite(norm_feature)] = 0
    with torch.no_grad():
        pred, prob = model(torch.tensor(norm_feature.astype(np.float32))[None, ...])
        pred = pred.numpy()[0].squeeze()
        prob = prob.sigmoid()[0].numpy().squeeze()
    return pred, prob


def decode_pred(pred, curr_ret1m, ewm_alpha, ewm_mean, ewm_var, label_std, valid, use_vola):
    if use_vola:
        curr_ret1m[~np.isfinite(curr_ret1m)] = 0
        curr_ret1m = curr_ret1m.squeeze()
        new_ewm_mean = ewm_alpha * curr_ret1m + (1 - ewm_alpha) * ewm_mean
        new_ewm_var = (
            ewm_alpha * ((curr_ret1m - new_ewm_mean) ** 2) + (1 - ewm_alpha) * ewm_var
        )
        ewm_mean[valid] = new_ewm_mean[valid]
        ewm_var[valid] = new_ewm_var[valid]
        curr_vola = np.log1p(ewm_var**0.5)
    else:
        curr_vola = 1

    pred = label_std * pred * curr_vola
    pred = np.expm1(np.abs(pred)) * np.sign(pred)
    pred[~valid] = 0
    return pred, ewm_mean, ewm_var
