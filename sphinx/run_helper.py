import time
import os
import sys
from typing import List, Tuple
from datetime import datetime
from argparse import ArgumentParser
import torch
import mosek
import psutil
import numpy as np
import pandas as pd

from sphinx.core.model import gen_model
from mona.common.logging import get_logger
from .okx_5min_sdk import SDKWrapper

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
    LOGGER.info("add %s to syspath", patch_path_s)


def check_load_model(cfg):
    model_name = 'model'
    checkpoints = sorted(os.listdir(cfg[model_name]["path"]))
    for model_dir_p in checkpoints:
        mdl_path = os.path.join(cfg[model_name]["path"], model_dir_p, f"{cfg[model_name]['epoch']}.pth.tar")
        torch.load(mdl_path, map_location="cpu", weights_only=False)
        LOGGER.info("load checkpoint:%s", str(mdl_path))


def check_load_mosek_license():
    os.environ["MOSEKLM_LICENSE_FILE"] = "mosek.lic"
    env = mosek.Env()
    # env.putlicensedebug(1)
    # env.checkoutlicense(mosek.feature.ptopt)
    version = env.getversion()
    LOGGER.info("mosek license loaded, version:%s", str(version))


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


def print_open_file():
    current_process = psutil.Process()
    file_handles = current_process.open_files()
    LOGGER.info("普通文件句柄数量: %s", str(len(file_handles)))


def okx5m_inst_feature():
    # 暂时 hard code, 顺序不能变！！！
    inst_feature_name = [
        "ret1m",  # 必须位于第一个！还会用于 vola 计算
        "ret10m",
        "ret40m",
        "ret80m",
        "ret160m",
        "midret1m",
        "midret10m",
        "midret40m",
        "midret80m",
        "midret160m",
        "twret1m",
        "twret10m",
        "twret40m",
        "twret80m",
        "twret160m",
        "twdiff_r1",
        "twdiff_r10",
        "twdiff_r40",
        "twdiff_r80",
        "twdiff_r160",
        "norm_ret1m_r40",
        "norm_ret1m_r80",
        "norm_ret1m_r160",
        "imbhl_r10",
        "imbhl_r40",
        "imbhl_r80",
        "imbhl_r160",
        "hldiff_r10",
        "hldiff_r40",
        "hldiff_r80",
        "hldiff_r160",
        "imb_d1_r1",
        "imb_d1_r10",
        "imb_d1_r40",
        "imb_d1_r160",
    ]
    return inst_feature_name


def bn5m_inst_feature():
    aug_feature_name = [
        "twdiff_r1_r10",
        "twdiff_r1_r40",
        "twdiff_r1_r80",
        "twdiff_r1_r160",
        "twdiff_r10_r40",
        "twdiff_r10_r80",
        "twdiff_r10_r160",
        "twdiff_r40_r80",
        "twdiff_r40_r160",
        "twdiff_r80_r160",
    ]
    return okx5m_inst_feature() + aug_feature_name


def base_mkt_feature():
    market_feature_name = [
        # "ret1m",
        "imbhl_r10",
        "imbhl_r40",
        "imbhl_r80",
        "imbhl_r160",
    ]
    return market_feature_name


def okx_mkt_feature():
    aug_mkt_feature = [
        "imb_1bp",
        "imb_1bp_norm60",
        "imb_1bp_norm360",
        "imb_1bp_dmean60",
        "imb_1bp_dmean360",
        "imb_10bp",
        "imb_10bp_norm60",
        "imb_10bp_norm360",
        "imb_10bp_dmean60",
        "imb_10bp_dmean360",
        "imb_100bp",
        "imb_100bp_norm60",
        "imb_100bp_norm360",
        "imb_100bp_dmean60",
        "imb_100bp_dmean360",
    ]
    return base_mkt_feature() + aug_mkt_feature


def v0_min_infer(task_tm_s, cfg, univ, model_name, model_id, sdk: SDKWrapper, inst_feature_name, market_feature_name) -> Tuple[List[pd.Series], List[pd.Series]]:
    # read history start timer
    st_time = time.time()

    torch.set_num_threads(1)
    horizon = cfg[model_name]["horizon"]
    channel = cfg[model_name]["channel"]
    use_vola = cfg[model_name]["use_vola"]
    assert channel == len(horizon)
    checkpoints = sorted(os.listdir(cfg[model_name]["path"]))
    checkpoint = torch.load(os.path.join(cfg[model_name]["path"], checkpoints[model_id], f"{cfg[model_name]['epoch']}.pth.tar"), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    all_feature_name_list = config["dataset"]["alphas"]
    assert all_feature_name_list[0] == "ret1m"

    # alpha_name_list = config["dataset"]["alphas"]
    model = gen_model(**config["model"], seq_len=config["dataset"]["seq_len"])
    model.load_state_dict(checkpoint[cfg[model_name]["state_dict_key"]])
    normalizer = checkpoint["normalizer"]

    market_list = [
        # f"market_universe_t60r1_oi_list30d_okx_futures",
        f"market_{cfg['universe']}",
        # "market_T30R20_r20to",
        # "market_T30R20_minuteto",
        # "market_T30R20_r20logto",
        # "market_T30R20_minutelogto",
    ]

    history_inst_self_feature_list = []
    for inst in univ:
        f_l = [sdk.deploy_read_history_alpha(task_tm_s, inst, f).rename(f) for f in inst_feature_name]
        history_inst_self_feature_list.append(pd.concat(f_l, axis=1))

    history_market_feature = [sdk.deploy_read_history_alpha(task_tm_s, market, "ret1m").rename(f"{market}_ret1m") for market in market_list]
    history_market_feature += [sdk.deploy_read_history_alpha(task_tm_s, market_list[0], alpha).rename(f"{market_list[0]}_{alpha}") for alpha in market_feature_name]
    history_market_feature = pd.concat(history_market_feature, axis=1)
    history_index = history_market_feature.index
    
    history_inst_feature_list = []
    for inst_feature in history_inst_self_feature_list:
        history_inst_feature_list.append(pd.concat([inst_feature, history_market_feature], axis=1)[all_feature_name_list].values.T)

    # Universe X Feature X T(1023)
    history_inst_feature = np.array(history_inst_feature_list)
     # 第一个 feature 必须是 ret1m！
    history_inst_ret1m = history_inst_feature[:, 0:1, :] 
    history_inst_ret1m = history_inst_ret1m[..., -(model.seq_len - 1):]

    norm_history_inst_feature = (history_inst_feature / normalizer["feature"].std[None, :, None]).clip(-normalizer["feature"].clip, normalizer["feature"].clip)
    norm_history_inst_feature = norm_history_inst_feature[..., -(model.seq_len - 1):]
    
    LOGGER.info(f"[{model_id}] read history cost {(time.time() - st_time)*1e3:.2f}ms")

    valid = pd.concat([sdk.deploy_read_history_alpha(task_tm_s, inst, "ret1m").notna().rename("valid") for inst in univ], axis=1).values
 
    # 设置状态, 训练集 gen_data 里有 max(10, h), 这里做检查
    assert all([h >= 10 for h in horizon])
    
    horizon = [max(10, h) for h in horizon]
    ewm_halflife = [h * 2 for h in horizon]
    ewm_alpha = [1 - np.exp(-np.log(2) / h) for h in ewm_halflife]
    ewm_mean = [np.zeros(len(univ)) for _ in horizon]
    ewm_var = [np.zeros(len(univ)) for _ in horizon]

    model.eval()
    model.reset_eval_model("incremental_eval")
    model.reset_cache_model()
    preds = [[] for _ in horizon]
    probs = [[] for _ in horizon]
    assert norm_history_inst_feature.shape[-1] == model.seq_len - 1, f"seq_len mismatch: {norm_history_inst_feature.shape[-1]} vs {model.seq_len - 1}"
    
    for i in range(norm_history_inst_feature.shape[-1]):
        encode_pred, encode_prob = gen_minute_encode_pred(model, norm_history_inst_feature[..., i:i + 1])
        for c in range(channel):
            pred, ewm_mean[c], ewm_var[c] = decode_pred(
                encode_pred[:, c],
                history_inst_ret1m[..., i:i + 1],
                ewm_alpha[c],
                ewm_mean[c],
                ewm_var[c],
                normalizer["label"].std[c:c + 1],
                valid[i],
                use_vola,
            )
            preds[c].append(pred)
            prob = encode_prob[:, c] * 2 - 1
            probs[c].append(prob)
    # hist_df = pd.DataFrame(preds, index=history_index[-(model.seq_len - 1):], columns=universe)

    LOGGER.info("ready to block on reading lastest data")

    # 读最新数据
    read_last_st = time.time()
    last_inst_self_feature_list = []
    for inst in univ:
        df_l = [sdk.deploy_read_last_alpha(task_tm_s, inst, name).rename(name) for name in inst_feature_name]
        last_inst_self_feature_list.append(pd.concat(df_l, axis=1))

    last_market_feature = [sdk.deploy_read_last_alpha(task_tm_s, market, "ret1m").rename(f"{market}_ret1m") for market in market_list]
    last_market_feature += [sdk.deploy_read_last_alpha(task_tm_s, market_list[0], alpha).rename(f"{market_list[0]}_{alpha}") for alpha in market_feature_name]
    last_market_feature = pd.concat(last_market_feature, axis=1)
    last_index = last_market_feature.index

    last_inst_feature = []
    for inst_feature in last_inst_self_feature_list:
        last_inst_feature.append(pd.concat([inst_feature, last_market_feature], axis=1)[all_feature_name_list].values.T)

    # Universe X Feature X T(1)
    last_inst_feature = np.array(last_inst_feature)
    # 第一个 feature 必须是 ret1m！
    last_inst_ret1m = last_inst_feature[:, 0:1, :]
    norm_last_inst_feature = (last_inst_feature / normalizer["feature"].std[None, :, None]).clip(-normalizer["feature"].clip, normalizer["feature"].clip)

    last_valid = np.array([sdk.deploy_read_last_alpha(task_tm_s, inst, "ret1m").notna().values[0] for inst in univ])
    
    LOGGER.info("read lastest data done")

    last_encode_pred, last_encode_prob = gen_minute_encode_pred(model, norm_last_inst_feature)
    last_preds = []
    last_probs = []
    for c in range(channel):
        last_pred, last_ewm_mean, last_ewm_var = decode_pred(
            last_encode_pred[:, c],
            last_inst_ret1m,
            ewm_alpha[c],
            ewm_mean[c],
            ewm_var[c],
            normalizer["label"].std[c:c + 1],
            last_valid,
            use_vola,
        )
        last_preds.append(last_pred)
        last_prob = last_encode_prob[:, c] * 2 - 1
        last_probs.append(last_prob)
    last_preds = [pd.Series(p, index=univ) for p in last_preds]
    last_probs = [pd.Series(p, index=univ) for p in last_probs]
    # print(f"[{model_id}] model cost {(time.time() - st_time)*1e3:.2f}ms")
    LOGGER.info(f"[{model_id}] read last cost {(time.time() - read_last_st)*1e3:.2f}ms")

    return last_preds, last_probs
