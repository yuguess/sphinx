import os
import sys
import traceback
import json5
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Mapping

from sphinx.util.runtime_config import OKX_EXCHANGE, CF_EXCHANGE, FREQ_1S, FREQ_5MIN, FREQ_1H
from sphinx.util.runtime_config import get_env_exchange, get_env_freq
from sphinx.util.path_utils import require_supported_runtime, deep_data_root, config_dir
from sphinx.util.exchange_api import get_dates, today_all_inst, read_universe, read_alpha, read_market_data, next_date


def parse_args():
    parser = ArgumentParser(description="strategy-dl data generation")
    parser.add_argument("horizon", type=int)
    parser.add_argument("-i", "--interval", required=True)
    parser.add_argument("-p", "--pool_num", type=int, default=24)
    parser.add_argument("-l", "--label", action="store_true")
    parser.add_argument("-u", "--universe", type=str, default="universe_t55r1_oi_CF")
    parser.add_argument("--label-start", type=int, default=0)
    parser.add_argument("--label-name", type=str, default=None)
    return parser.parse_args()


args = parse_args()
require_supported_runtime()


HORIZON = args.horizon
LABEL_START = args.label_start
DEEP_DATA_DIR = str(deep_data_root() / args.universe)
POOL_NUM = args.pool_num
BEGIN_DATE, END_DATE = args.interval.split("/")


print(f"DEEP_DATA_DIR:{DEEP_DATA_DIR}")

def prev_date_num() -> int:
    exchange = get_env_exchange()
    if exchange == OKX_EXCHANGE:
        return 5
    elif exchange == CF_EXCHANGE:
        return 8
    else:
        raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")


PREV_DATE_NUM = prev_date_num()

with open(config_dir() / "alpha_cfg.json5", "r") as f:
    alpha_cfg = json5.load(f)

MARKET_ALPHAS = [
    "ret1m",
    "imbhl_r10",
    "imbhl_r40",
    "imbhl_r80",
    "imbhl_r160",
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


def inc_ewmstd(df, halflife):
    assert isinstance(df, pd.Series)
    assert halflife > 0
    alpha = 1 - np.exp(-np.log(2) / halflife)
    values = df.values
    mean = np.zeros_like(values) * np.nan
    var = np.zeros_like(values) * np.nan

    for i, value in enumerate(values):
        if i == 0:
            mean[i] = value
            var[i] = 0
        elif i != 0:
            mean[i] = value * alpha + mean[i - 1] * (1 - alpha)
            var[i] = ((value - mean[i])**2) * alpha + var[i - 1] * (1 - alpha)
        if np.isnan(mean[i]):
            mean[i] = mean[i - 1]
            var[i] = var[i - 1]
    return pd.Series(var**0.5, index=df.index)


def gen_inst_dates(date, today_all_insts, prev_and_next_univs):
    valid_insts = set(today_all_insts)
    ret = set()
    for univ in prev_and_next_univs:
        ret |= set(univ) & valid_insts
    return ret


def build_inst_dates():
    all_dates = get_dates(BEGIN_DATE, END_DATE)
    with Pool(POOL_NUM) as p:
        all_inst_list = p.map(today_all_inst, all_dates)
    with Pool(POOL_NUM) as p:
        all_univ_list = p.starmap(read_universe, [(date, args.universe) for date in all_dates])
    all_univ_list = [u.index for u in all_univ_list]

    inst_dates = {}
    task_list = []
    for date_idx, date in enumerate(all_dates):
        univs = []
        for i in range(date_idx - 1, date_idx + PREV_DATE_NUM):
            if 0 <= i < len(all_univ_list):
                univs.append(all_univ_list[i])
        task_list.append((date, all_inst_list[date_idx], univs))
    with Pool(POOL_NUM) as p:
        date_inst_list = p.starmap(gen_inst_dates, task_list)
    for i, date_inst in enumerate(date_inst_list):
        for inst in date_inst:
            inst_dates.setdefault(inst, []).append(all_dates[i])
    return inst_dates


def gen_alpha(alpha_name, dates, inst, valid, dataset_prefix, output_name=None):
    alpha = pd.concat([read_alpha(date, inst, alpha_name) for date in dates], axis=0)[valid]
    assert (~np.isfinite(alpha)).sum() == 0
    alpha.rename(alpha_name).to_frame().to_pickle(f"{dataset_prefix}/{output_name or alpha_name}.pkl.zip")


def concat_kline(dates, inst, field):
    return pd.concat([read_market_data(date, inst, f"kline/{field}") for date in dates], axis=0)


def concat_midprice(dates, inst):
    return pd.concat([read_market_data(date, inst, "quotes/midprice") for date in dates], axis=0)


def label_scale_name(label_name):
    digits = "".join([i for i in label_name if i.isdigit()])
    if digits:
        return f"vola{digits}m"
    raise ValueError(f"label_name must include digits for vola lookup: {label_name}")


def default_return_label_name():
    freq = get_env_freq()
    if freq == FREQ_1S:
        if LABEL_START == 0:
            return f"mid_return{HORIZON}s"
        elif LABEL_START > 0:
            return f"mid_return{LABEL_START}s{HORIZON}s"
        else:
            raise ValueError(f"label_start must be non-negative, got {LABEL_START}")
    elif freq in {FREQ_5MIN, FREQ_1H}:
        return f"return{HORIZON}m"
    else:
        raise ValueError(f"unsupported FREQ={freq!r}")


def build_labels(close, volume, turnover, ret1m):
    freq = get_env_freq()
    label_name = args.label_name or default_return_label_name()
    if LABEL_START >= HORIZON:
        raise ValueError(f"label_start must be smaller than horizon, got {LABEL_START} >= {HORIZON}")

    tmp_horizon = max(10, HORIZON)
    rtn_vola = inc_ewmstd(ret1m.fillna(0), tmp_horizon * 2)
    rtn_vola.iloc[:tmp_horizon] = 0

    if freq == FREQ_1S:
        label_return = close.shift(-HORIZON) / close.shift(-LABEL_START) - 1
        label_return.iloc[-HORIZON:] = np.nan
        label_vwreturn = label_return
    elif freq in {FREQ_5MIN, FREQ_1H}:
        label_return = close.shift(-HORIZON) / close - 1
        end_price = (turnover.rolling(HORIZON + 1).sum() / volume.rolling(HORIZON + 1).sum()).shift(-HORIZON - 1)
        end_price = end_price.where(np.isfinite(end_price), np.nan).ffill()
        begin_price = (turnover / volume).shift(-1)
        begin_price = begin_price.where(np.isfinite(begin_price), np.nan).ffill()
        label_vwreturn = end_price / begin_price - 1
        label_vwreturn.iloc[-HORIZON:] = np.nan
    else:
        raise ValueError(f"unsupported FREQ={freq!r}")
    return label_name, rtn_vola, label_return, label_vwreturn


def gen_data(data):
    try:
        inst, dates = data
        print(f"BEGIN {inst}: {dates[0]} to {dates[-1]}, {len(dates)} days")
        exchange = get_env_exchange()
        if exchange == OKX_EXCHANGE:
            valid = pd.concat([read_alpha(date, inst, "ret1m") for date in dates], axis=0).abs().fillna(0) > -1
        elif exchange == CF_EXCHANGE:
            valid = pd.concat([read_alpha(date, inst, "valid") for date in dates], axis=0) == 1
        else:
            raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")
        freq = get_env_freq()
        if freq == FREQ_1S:
            all_close = concat_midprice(dates, inst)
        elif freq in {FREQ_5MIN, FREQ_1H}:
            all_close = concat_kline(dates, inst, "close")
        else:
            raise ValueError(f"unsupported FREQ={freq!r}")
        close = all_close[valid]
        volume = concat_kline(dates, inst, "volume")[valid]
        turnover = concat_kline(dates, inst, "turnover")[valid]
        assert np.isfinite(close).all()
        assert np.isfinite(volume).all()
        assert np.isfinite(turnover).all()

        ret1m = pd.concat([read_alpha(date, inst, "ret1m") for date in dates], axis=0)[valid]
        assert (~np.isfinite(ret1m)).sum() == 0

        dataset_prefix = f"{DEEP_DATA_DIR}/{inst}_{dates[0]}"
        os.makedirs(f"{dataset_prefix}/label", exist_ok=True)
        os.makedirs(f"{dataset_prefix}/feature", exist_ok=True)
        print(f"create dataset_prefix f{dataset_prefix}")

        label_name, rtn_vola, label_return, label_vwreturn = build_labels(close, volume, turnover, ret1m)

        rtn_vola.rename("vola").to_frame().to_pickle(f"{dataset_prefix}/label/{label_scale_name(label_name)}.pkl.zip")
        label_return.rename(f"label_{label_name}").to_frame().to_pickle(f"{dataset_prefix}/label/{label_name}.pkl.zip")
        if freq in {FREQ_5MIN, FREQ_1H}:
            label_vwreturn.rename(f"label_vwreturn{HORIZON}m").to_frame().to_pickle(f"{dataset_prefix}/label/vwreturn{HORIZON}m.pkl.zip")
        elif freq == FREQ_1S:
            pass
        else:
            raise ValueError(f"unsupported FREQ={freq!r}")
        turnover.rename("turnover").to_frame().to_pickle(f"{dataset_prefix}/label/turnover.pkl.zip")
        exchange = get_env_exchange()
        if exchange == CF_EXCHANGE:
            label_return115m = all_close.shift(-115) / close - 1
            label_return115m.reindex(label_return.index).rename("label_return115m").to_frame().to_pickle(f"{dataset_prefix}/label/return115m.pkl.zip")
        elif exchange == OKX_EXCHANGE:
            pass
        else:
            raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")

        if args.label:
            samples = (label_vwreturn.notna() & rtn_vola.gt(0)).sum()
            print(f"DONE {inst}: {dates[0]} to {dates[-1]}, {len(dates)} days, {samples} samples")
            return samples

        for alpha in alpha_cfg["alphas"]:
            gen_alpha(alpha, dates, inst, valid, f"{dataset_prefix}/feature")
        exchange = get_env_exchange()
        if exchange == OKX_EXCHANGE and freq == FREQ_5MIN:
            market_name = f"market_{args.universe}"
            for alpha in MARKET_ALPHAS:
                gen_alpha(alpha, dates, market_name, valid, f"{dataset_prefix}/feature", f"{market_name}_{alpha}")
        elif exchange == OKX_EXCHANGE and freq in {FREQ_1S, FREQ_1H}:
            pass
        elif exchange == CF_EXCHANGE:
            pass
        else:
            raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")

        samples = (label_vwreturn.notna() & rtn_vola.gt(0)).sum()
        print(f"DONE {inst}: {dates[0]} to {dates[-1]}, {len(dates)} days, {samples} samples")
        return samples
    except Exception as exc:
        print(f"=== ERROR {data[0]}: {exc}", traceback.format_exc())
        return None


def gen_task(inst, dates):
    insert_dates = sorted(set(dates))
    date_lists = []
    tmp = [insert_dates[0]]
    for date in insert_dates[1:]:
        ndate = next_date(tmp[-1])
        if date == ndate:
            tmp.append(date)
        elif date != ndate:
            date_lists.append(tmp)
            tmp = [date]
    date_lists.append(tmp)
    return [(inst, dates) for dates in date_lists]


def arrange_tasks(tasks):
    tasks = sorted(tasks, key=lambda x: len(x[1]), reverse=True)
    task_list = [[] for _ in range(POOL_NUM)]
    task_list_len = np.zeros(POOL_NUM)
    for task in tasks:
        idx = task_list_len.argmin()
        task_list[idx].append(task)
        task_list_len[idx] += len(task[1])
    return task_list


def do_gen_data(tasks):
    ret = 0
    for task in tasks:
        samples = gen_data(task)
        if samples is not None:
            ret += samples
    return ret


inst_dates = build_inst_dates()
with Pool(POOL_NUM) as p:
    tasks = p.starmap(gen_task, inst_dates.items())
tasks = [task for grouped in tasks for task in grouped]
tasks = arrange_tasks(tasks)

for i in range(POOL_NUM):
    print(f"pool {i}: {len(tasks[i])} tasks, {sum(len(x[1]) for x in tasks[i])} days")

if not args.label:
    print("del data?")
    import time

    time.sleep(3)
    os.system(f"rm -rf {DEEP_DATA_DIR}")

with Pool(POOL_NUM) as p:
    ret = p.map(do_gen_data, tasks)

sample_num = sum(x for x in ret if x is not None)
print(f"total sample num: {sample_num / 1e4:.2f}w")
