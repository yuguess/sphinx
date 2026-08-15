import os
import time
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sphinx.util.exchange_api import get_env_exchange, today_all_inst, get_dates, read_universe, prev_date, next_date, read_alpha
from sphinx.util.exchange_api import read_basedata
from sphinx.util.path_utils import deep_data_root
from sphinx.base_adt import SymS, DateS_L
from mona.common.logging import get_logger

SymDatesTp = Tuple[SymS, DateS_L]
SymDatesTp_L = List[SymDatesTp]


LG = get_logger('gen_data')


alpha_lists = {
    "all": [
        "ret1m",
        "ret10m",
        "ret40m",
        "ret80m",
        "ret160m",
        # "midret1m",
        # "midret10m",
        # "midret40m",
        # "midret80m",
        # "midret160m",
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
        "norm_ret1m_r40",
        "norm_ret1m_r80",
        "norm_ret1m_r160",
        "hldiff_r10",
        "hldiff_r40",
        "hldiff_r80",
        "hldiff_r160",
        "imbhl_r10",
        "imbhl_r40",
        "imbhl_r80",
        "imbhl_r160",
    ],
    "ret1m": ["ret1m"],
    "empty": [],
}

market_alphas = [
    "imbhl_r10",
    "imbhl_r40",
    "imbhl_r80",
    "imbhl_r160",
]


def evenly_split_task(tasks: List[Tuple[SymS, DateS_L]], pool_num: int):
    tasks = sorted(tasks, key=lambda x: len(x[1]), reverse=True)
    task_list = [[] for _ in range(pool_num)]
    task_list_len = np.zeros(pool_num)
    for task in tasks:
        idx = task_list_len.argmin()
        task_list[idx].append(task)
        task_list_len[idx] += len(task[1])
    return task_list


def inc_ewmstd(df, halflife):
    assert isinstance(df, pd.Series)
    assert halflife > 0
    alpha = 1 - np.exp(-np.log(2) / halflife)
    df_index = df.index
    df = df.values
    mean = np.zeros_like(df) * np.nan
    var = np.zeros_like(df) * np.nan

    for i, _ in enumerate(df):
        if i == 0:
            mean[i] = df[i]
            var[i] = 0
        else:
            mean[i] = df[i] * alpha + mean[i - 1] * (1 - alpha)
            var[i] = ((df[i] - mean[i])**2) * alpha + var[i - 1] * (1 - alpha)
        if np.isnan(mean[i]):
            mean[i] = mean[i - 1]
            var[i] = var[i - 1]
    return pd.Series(var**0.5, index=df_index)


def gen_alpha(alpha_name, dates, inst, valid, dataset_prefix):
    alpha = pd.concat([read_alpha(date, inst, alpha_name) for date in dates], axis=0)[valid]
    assert (~np.isfinite(alpha)).sum() == 0
    alpha.rename(alpha_name).to_frame().to_pickle(f"{dataset_prefix}/feature/{alpha_name}.pkl.zip")
    
    
def gen_task(inst, dates) -> List[Tuple[SymS, DateS_L]]:
    # # 如果 inst 的日期不连续，且仅差了一天，那么插入这一天
    # next_dates = [next_date(date) for date in dates]
    # prev_dates = [prev_date(date) for date in dates]
    # insert_dates = sorted(set(dates + next_dates + prev_dates))
    # # dev 保证了处于 universe 的后一天是有数据的，但是交易所历史数据可能缺失，所以增加 today all inst 的判断
    # insert_dates = [date for date in insert_dates if date >= BEGIN_DATE and date <= END_DATE and inst in today_all_inst(date)]

    insert_dates = sorted(set(dates))
    # 此时如果数据依然不连续就应该分段了
    date_lists = []
    tmp = [insert_dates[0]]
    for date in insert_dates[1:]:
        ndate = next_date(tmp[-1])
        if date == ndate:
            tmp.append(date)
        else:
            date_lists.append(tmp)
            tmp = [date]
    date_lists.append(tmp)
    return [(inst, d) for d in date_lists]


def gen_data(data: SymDatesTp, xhg, args, deep_data_dir, alphas):
    horizon = args.horizon
    residual = args.residual
    inst, dates = data
    LG.info(f"BEGIN {inst}: {dates[0]} to {dates[-1]}, {len(dates)} days")
    
    try:
        if xhg == "CF":
            valid = pd.concat([read_alpha(date, inst, "valid") for date in dates], axis=0) == 1
        else:
            valid = pd.concat([read_alpha(date, inst, "ret1m") for date in dates], axis=0).abs().fillna(0) > -1

        basedata = [read_basedata(date, inst) for date in dates]

        close = pd.concat([bd["close"] for bd in basedata], axis=0)[valid]
        volume = pd.concat([bd["volume"] for bd in basedata], axis=0)[valid]
        turnover = pd.concat([bd["turnover"] for bd in basedata], axis=0)[valid]
        assert np.isfinite(close).all()
        assert np.isfinite(volume).all()
        assert np.isfinite(turnover).all()

        # ret1m = pd.concat([read_alpha(date, inst, f"res_{args.residual}_ret1m") for date in dates], axis=0)
        ret1m = pd.concat([read_alpha(date, inst, "ret1m") for date in dates], axis=0)[valid]
        # vwap_slippage = pd.concat([read_alpha(date, inst, "vwap_slippage") for date in dates], axis=0)[valid]
        if residual is not None:
            market_name = f"market_{args.universe}_{residual}"
            market_ret1m = pd.concat([read_alpha(date, market_name, "ret1m") for date in dates], axis=0)[valid]
            ret1m = ret1m - market_ret1m

        assert (~np.isfinite(ret1m)).sum() == 0
        # assert (~np.isfinite(vwap_slippage)).sum() == 0

        # mkdir
        dataset_prefix = f"{deep_data_dir}/{inst}_{dates[0]}"
        os.makedirs(f"{dataset_prefix}/label", exist_ok=True)
        os.makedirs(f"{dataset_prefix}/feature", exist_ok=True)

        # vola
        rtn_vola = ret1m.copy()  # TODO：做期货的时候注意这里，隔夜等
        rtn_vola = rtn_vola.fillna(0)
        rtn_vola = inc_ewmstd(rtn_vola, horizon * 2)
        rtn_vola.iloc[:horizon] = 0

        # label return
        # label_return = ret1m.rolling(HORIZON).sum().shift(-HORIZON)
        # TODO: label 应该在 clean basedata 里读，这样有 nan
        label_return = close.shift(-horizon) / close - 1

        # label_vwreturn = label_return - vwap_slippage.shift(-1)
        end_price = (turnover.rolling(horizon - 1).sum() / volume.rolling(horizon - 1).sum()).shift(-horizon)
        end_price = end_price.where(np.isfinite(end_price), np.nan).ffill()
        begin_price = (turnover / volume).shift(-1)
        begin_price = begin_price.where(np.isfinite(begin_price), np.nan).ffill()
        label_vwreturn = end_price / begin_price - 1
        label_vwreturn.iloc[-horizon:] = np.nan

        # dump_label
        label_prefix = "" if residual is None else f"res_{residual}_"
        rtn_vola.rename(f"vola").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}vola{horizon}m.pkl.zip")
        label_return.rename(f"label_{label_prefix}return{horizon}m").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}return{horizon}m.pkl.zip")
        # vwap_slippage.rename(f"label_vwap_slippage").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}vwap_slippage.pkl.zip")
        label_vwreturn.rename(f"label_{label_prefix}vwreturn{horizon}m").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}vwreturn{horizon}m.pkl.zip")
        turnover.rename("turnover").to_frame().to_pickle(f"{dataset_prefix}/label/turnover.pkl.zip")
        
        if args.label:
            return 1

        for alpha in alphas:
            gen_alpha(alpha, dates, inst, valid, dataset_prefix)

        if xhg != "CF" and xhg != "coinbase":
            market_name = f"market_{args.universe}"
            market_ret1m = pd.concat([read_alpha(date, market_name, "ret1m") for date in dates], axis=0)[valid]
            assert (~np.isfinite(market_ret1m)).sum() == 0
            market_ret1m.rename(market_name).to_frame().to_pickle(f"{dataset_prefix}/feature/{market_name}_ret1m.pkl.zip")

            for market_feature_name in market_alphas:
                market_feature = pd.concat([read_alpha(date, market_name, market_feature_name) for date in dates], axis=0)[valid]
                assert (~np.isfinite(market_feature)).sum() == 0
                market_feature.rename(market_feature_name).to_frame().to_pickle(f"{dataset_prefix}/feature/{market_name}_{market_feature_name}.pkl.zip")

        samples = (label_vwreturn.notna() & rtn_vola.gt(0)).sum()
        LG.info(f"DONE {inst}: {dates[0]} to {dates[-1]}, {len(dates)} days, {samples} samples")
        return samples
    except Exception as e:
        LG.info(f"=== ERROR {inst}: {e}", traceback.format_exc())
        return 0


def do_gen_data(tasks: SymDatesTp_L, xhg, args, deep_data_dir, alphas):
    ret = 0
    for task in tasks:
        ret += gen_data(task, xhg, args, deep_data_dir, alphas)
    return ret


def parse_args():
    parser = ArgumentParser(description='deep_crypto data generation')
    parser.add_argument('horizon', type=int, help='config file path')
    parser.add_argument('-a', '--alpha_list', type=str, default="all")
    # parser.add_argument('-i', '--interval', type=str, default="2020-01-01/2024-06-01")
    parser.add_argument('-i', '--interval', type=str)
    parser.add_argument('-r', '--residual', type=str)
    parser.add_argument('-p', '--pool_num', type=int, default=24)
    parser.add_argument('-l', '--label', action='store_true')
    # parser.add_argument('-r', '--residual', type=str, default="beta1_T20R20_r20to")
    # beta1_T10R20_r20to, beta1_T20R20_r20to, beta1_T30R20_r20to
    # beta1_T10R20_eq, beta1_T20R20_eq, beta1_T30R20_eq
    parser.add_argument('-u', '--universe', type=str)
    return parser.parse_args()


def build_up_inst_dates(all_inst_list, bg_date_s, ed_date_s, univ, xhg) -> Dict[SymS, DateS_L]:
    inst_dates = {}
    prev_prev_all_inst_list = [[]] + [[]] + all_inst_list[:-2]
    prev_all_inst_list = [[]] + all_inst_list[:-1]
    next_all_inst_list = all_inst_list[1:] + [[]]
    for date_idx, date in enumerate(get_dates(bg_date_s, ed_date_s)):
        universe = read_universe(date, univ).index
        pdate = prev_date(date)
        ppdate = prev_date(pdate)
        ndate = next_date(date)
        prev_all_inst = prev_all_inst_list[date_idx]
        prev_prev_all_inst = prev_prev_all_inst_list[date_idx]
        next_all_inst = next_all_inst_list[date_idx]
        for inst in universe:
            if inst not in inst_dates:
                inst_dates[inst] = []
            inst_dates[inst].append(date)
            if inst in prev_all_inst:
                inst_dates[inst].append(pdate)
            if inst in next_all_inst:
                inst_dates[inst].append(ndate)
            if xhg == "CF":
                if inst in prev_prev_all_inst:
                    inst_dates[inst].append(ppdate)
    return inst_dates


def main():
    args = parse_args()
    
    pool_num = args.pool_num
    alphas = alpha_lists[args.alpha_list]
    exchange = get_env_exchange()
    # deep_data_dir = f"{data_root_dir()}/deep/{exchange}/{args.universe}"
    deep_data_dir = str(deep_data_root() / args.universe)
    
    if not args.label:
        print(f"del data? {deep_data_dir}")
        time.sleep(3)
        os.system(f"rm -rf {deep_data_dir}")
    
    BEGIN_DATE, END_DATE = args.interval.split("/")
    if BEGIN_DATE <= "2020-01-01" and exchange == "okx":
        BEGIN_DATE = "2020-01-02"

    with Pool(pool_num) as p:
        all_inst_list = p.map(today_all_inst, get_dates(BEGIN_DATE, END_DATE))

    inst_to_dates = build_up_inst_dates(all_inst_list, BEGIN_DATE, END_DATE, args.universe, exchange)

    with Pool(pool_num) as p:
        tasks = p.starmap(gen_task, inst_to_dates.items())
    # each task is [(inst, datas), (inst, dates)]
    tasks = [x for y in tasks for x in y]

    tasks = evenly_split_task(tasks, pool_num)
    for i in range(pool_num):
        LG.info(f"pool {i}: {len(tasks[i])} tasks, {sum([len(x[1]) for x in tasks[i]])} days")

    with Pool(pool_num) as p:
        ret = p.starmap(do_gen_data, [(task_l, exchange, args, deep_data_dir, alphas) for task_l in tasks])

    sample_num = sum([x for x in ret if x is not None])
    LG.info(f"total sample num: {sample_num / 1e4:.2f}w")


if __name__ == "__main__":
    main()
