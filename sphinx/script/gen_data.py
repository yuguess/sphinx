import os
import numpy as np
import pandas as pd
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool

from ..exchange_api.exchange_api import EXCHANGE, data_root_dir, get_dates, next_date, prev_date, read_alpha, read_basedata, read_orderbook, read_universe, today_all_inst


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
    parser.add_argument('-u', '--universe', type=str, default="T20R20")
    return parser.parse_args()

args = parse_args()

HORIZON = args.horizon
RESIDUAL = args.residual
DEEP_DATA_DIR = f"{data_root_dir()}/deep/{EXCHANGE}/{args.universe}"
POOL_NUM = args.pool_num
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

alphas = alpha_lists[args.alpha_list]

if EXCHANGE == "coinbase":
    alphas.append("imb_d1_r1")
    alphas.append("imb_d1_r10")
    alphas.append("imb_d1_r40")
    alphas.append("imb_d1_r160")
    alphas.append("imb_d2_r1")
    alphas.append("imb_d2_r10")
    alphas.append("imb_d2_r40")
    alphas.append("imb_d2_r160")
    alphas.append("imb_d3_r1")
    alphas.append("imb_d3_r10")
    alphas.append("imb_d3_r40")
    alphas.append("imb_d3_r160")
    alphas.append("imb_d4_r1")
    alphas.append("imb_d4_r10")
    alphas.append("imb_d4_r40")
    alphas.append("imb_d4_r160")
    alphas.append("imb_d5_r1")
    alphas.append("imb_d5_r10")
    alphas.append("imb_d5_r40")
    alphas.append("imb_d5_r160")
    alphas.append("midret1m")
    alphas.append("midret10m")
    alphas.append("midret40m")
    alphas.append("midret80m")
    alphas.append("midret160m")

if EXCHANGE == "coinbase":
    assert args.universe in ["BTC", "BTCETH"]
if EXCHANGE == "coinbase" and (args.universe == "BTC" or args.universe == "BTCETH"):
    append_alphas = []
    for alpha in alphas:
        append_alphas.append(f"binance/{alpha}")
        append_alphas.append(f"spot/{alpha}")
        append_alphas.append(f"binance/spot/{alpha}")
    alphas = alphas + append_alphas
    # alphas = append_alphas
# res_alphas = [f"res_{args.residual}_{alpha}" for alpha in alphas]
# alphas = alphas + res_alphas

inst_dates = {}
# embed()
BEGIN_DATE, END_DATE = args.interval.split("/")
if BEGIN_DATE <= "2020-01-01" and EXCHANGE == "okx":
    BEGIN_DATE = "2020-01-02"
if BEGIN_DATE <= "2021-11-01" and EXCHANGE == "bybit":
    BEGIN_DATE = "2021-11-01"


with Pool(POOL_NUM) as p:
    all_inst_list = p.map(today_all_inst, get_dates(BEGIN_DATE, END_DATE))
prev_prev_all_inst_list = [[]] + [[]] + all_inst_list[:-2]
prev_all_inst_list = [[]] + all_inst_list[:-1]
next_all_inst_list = all_inst_list[1:] + [[]]
for date_idx, date in enumerate(get_dates(BEGIN_DATE, END_DATE)):
    universe = read_universe(date, args.universe).index
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
        if EXCHANGE == "CF":
            if inst in prev_prev_all_inst:
                inst_dates[inst].append(ppdate)


def gen_alpha(alpha_name, dates, inst, valid, dataset_prefix):
    alpha = pd.concat([read_alpha(date, inst, alpha_name) for date in dates], axis=0)[valid]
    assert (~np.isfinite(alpha)).sum() == 0
    alpha.rename(alpha_name).to_frame().to_pickle(f"{dataset_prefix}/feature/{alpha_name}.pkl.zip")


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


def gen_data(data, para):
    try:
        inst, dates = data
        # if not inst.startswith("MEW"):
        #     return 1
        print(f"BEGIN {inst}: {dates[0]} to {dates[-1]}, {len(dates)} days")

        if EXCHANGE == "CF":
            valid = pd.concat([read_alpha(date, inst, "valid") for date in dates], axis=0) == 1
        else:
            valid = pd.concat([read_alpha(date, inst, "ret1m") for date in dates], axis=0).abs().fillna(0) > -1
        
        basedata = [read_basedata(date, inst) for date in dates]

        close = pd.concat([bd["close"] for bd in basedata], axis=0)[valid]
        if EXCHANGE == "coinbase":
            orderbook = [read_orderbook(date, inst) for date in dates]
            mid_price = pd.concat([(ob["bid1_price"] + ob["ask1_price"]) / 2 for ob in orderbook], axis=0)[valid]
            assert np.isfinite(mid_price).all()
        volume = pd.concat([bd["volume"] for bd in basedata], axis=0)[valid]
        turnover = pd.concat([bd["turnover"] for bd in basedata], axis=0)[valid]
        assert np.isfinite(close).all()
        assert np.isfinite(volume).all()
        assert np.isfinite(turnover).all()

        # ret1m = pd.concat([read_alpha(date, inst, f"res_{args.residual}_ret1m") for date in dates], axis=0)
        ret1m = pd.concat([read_alpha(date, inst, "ret1m") for date in dates], axis=0)[valid]
        # vwap_slippage = pd.concat([read_alpha(date, inst, "vwap_slippage") for date in dates], axis=0)[valid]
        if RESIDUAL is not None:
            market_name = f"market_{args.universe}_{RESIDUAL}"
            market_ret1m = pd.concat([read_alpha(date, market_name, "ret1m") for date in dates], axis=0)[valid]
            ret1m = ret1m - market_ret1m
        assert (~np.isfinite(ret1m)).sum() == 0
        # assert (~np.isfinite(vwap_slippage)).sum() == 0

        # mkdir
        dataset_prefix = f"{DEEP_DATA_DIR}/{inst}_{dates[0]}"
        os.makedirs(f"{dataset_prefix}/label", exist_ok=True)
        os.makedirs(f"{dataset_prefix}/feature", exist_ok=True)
        if EXCHANGE == "coinbase":
            os.makedirs(f"{dataset_prefix}/feature/spot", exist_ok=True)
            os.makedirs(f"{dataset_prefix}/feature/binance", exist_ok=True)
            os.makedirs(f"{dataset_prefix}/feature/binance/spot", exist_ok=True)
            os.makedirs(f"{dataset_prefix}/label/spot", exist_ok=True)
            os.makedirs(f"{dataset_prefix}/label/binance", exist_ok=True)
            os.makedirs(f"{dataset_prefix}/label/binance/spot", exist_ok=True)

        # vola
        rtn_vola = ret1m.copy()  # TODO：做期货的时候注意这里，隔夜等
        rtn_vola = rtn_vola.fillna(0)
        # rtn_vola = rtn_vola.ewm(halflife=HORIZON * 2, min_periods=HORIZON * 1).std().ffill().fillna(0)
        # if EXCHANGE == "coinbase":
        #     rtn_vola = inc_ewmstd(rtn_vola, max(HORIZON * 2, 20))
        # else:
        #     rtn_vola = inc_ewmstd(rtn_vola, HORIZON * 2)
        rtn_vola = inc_ewmstd(rtn_vola, HORIZON * 2)
        rtn_vola.iloc[:HORIZON] = 0

        # label return
        # label_return = ret1m.rolling(HORIZON).sum().shift(-HORIZON)
        # TODO: label 应该在 clean basedata 里读，这样有 nan
        label_return = close.shift(-HORIZON) / close - 1
        if EXCHANGE == "coinbase":
            label_midreturn = mid_price.shift(-HORIZON) / mid_price - 1
            # label_bn_midreturn1m = pd.concat([read_alpha(date, inst, "binance/midret1m") for date in dates], axis=0)[valid].shift(-1)

        # label_vwreturn = label_return - vwap_slippage.shift(-1)
        end_price = (turnover.rolling(HORIZON - 1).sum() / volume.rolling(HORIZON - 1).sum()).shift(-HORIZON)
        end_price = end_price.where(np.isfinite(end_price), np.nan).ffill()
        begin_price = (turnover / volume).shift(-1)
        begin_price = begin_price.where(np.isfinite(begin_price), np.nan).ffill()
        label_vwreturn = end_price / begin_price - 1
        label_vwreturn.iloc[-HORIZON:] = np.nan

        # dump_label
        label_prefix = "" if RESIDUAL is None else f"res_{RESIDUAL}_"
        rtn_vola.rename(f"vola").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}vola{HORIZON}m.pkl.zip")
        label_return.rename(f"label_{label_prefix}return{HORIZON}m").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}return{HORIZON}m.pkl.zip")
        # vwap_slippage.rename(f"label_vwap_slippage").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}vwap_slippage.pkl.zip")
        label_vwreturn.rename(f"label_{label_prefix}vwreturn{HORIZON}m").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}vwreturn{HORIZON}m.pkl.zip")
        if EXCHANGE == "coinbase":
            label_midreturn.rename(f"label_{label_prefix}midreturn{HORIZON}m").to_frame().to_pickle(f"{dataset_prefix}/label/{label_prefix}midreturn{HORIZON}m.pkl.zip")
            # label_bn_midreturn1m.rename(f"label_{label_prefix}binance_midreturn1m").to_frame().to_pickle(f"{dataset_prefix}/label/binance/{label_prefix}midreturn1m.pkl.zip")
        # label minmaxdiff
        # label_minmaxdiff = []
        # for h in range(1, HORIZON + 1):
        #     label_minmaxdiff.append((close.shift(-h) / close - 1).rename(f"return{h}m"))
        # label_minmaxdiff = pd.concat(label_minmaxdiff, axis=1)
        # label_min = label_minmaxdiff.min(axis=1)
        # label_max = label_minmaxdiff.max(axis=1)
        # label_minmaxdiff = label_max + label_min
        # label_minmaxdiff.rename(f"label_minmaxdiff{HORIZON}m").to_frame().to_pickle(f"{dataset_prefix}/label/minmaxdiff{HORIZON}m.pkl.zip")

        if args.label:
            return 1

        # feature
        # def gen_alpha(alpha_name):
        #     alpha = pd.concat([read_alpha(date, inst, alpha_name) for date in dates], axis=0)[valid]
        #     assert (~np.isfinite(alpha)).sum() == 0
        #     alpha.rename(alpha_name).to_frame().to_pickle(f"{dataset_prefix}/feature/{alpha_name}.pkl.zip")

        if para:
            assert EXCHANGE == "coinbase"
            assert args.universe == "BTC"
            with Pool(POOL_NUM) as p:
                p.starmap(gen_alpha, [(alpha, dates, inst, valid, dataset_prefix) for alpha in alphas])
        else:
            for alpha in alphas:
                gen_alpha(alpha, dates, inst, valid, dataset_prefix)

        if EXCHANGE != "CF" and EXCHANGE != "coinbase":

            # market ret1m
            for market_suffix in [
                "r20to",
                # "minuteto",
                "r20logto",
                # "minutelogto",
            ]:
                market_name = f"market_{args.universe}_{market_suffix}"
                market_ret1m = pd.concat([read_alpha(date, market_name, "ret1m") for date in dates], axis=0)[valid]
                assert (~np.isfinite(market_ret1m)).sum() == 0
                # TODO: wrong name
                market_ret1m.rename(market_name).to_frame().to_pickle(f"{dataset_prefix}/feature/{market_name}_ret1m.pkl.zip")
                # for i in [1, 10, 40, 80, 160]:
                #     market_alpha = pd.concat([read_alpha(date, market_name, f"ret{i}m") for date in dates], axis=0)
                #     assert market_alpha.isna().sum() == 0
                #     market_alpha.rename(f"{market_name}_ret{i}m").to_frame().to_pickle(f"{dataset_prefix}/feature/{market_name}_ret{i}m.pkl.zip")

            # market feature
            for market_name in [
                f"market_{args.universe}_r20to",
                # f"market_{args.universe}_r20logto",
                # f"market_{args.universe}_minuteto",
                # f"market_{args.universe}_minutelogto",
                # f"market_{args.universe}_median",
            ]:
                for market_feature_name in market_alphas:
                    market_feature = pd.concat([read_alpha(date, market_name, market_feature_name) for date in dates], axis=0)[valid]
                    assert (~np.isfinite(market_feature)).sum() == 0
                    # TODO: wrong name
                    market_feature.rename(market_feature_name).to_frame().to_pickle(f"{dataset_prefix}/feature/{market_name}_{market_feature_name}.pkl.zip")

        samples = (label_vwreturn.notna() & rtn_vola.gt(0)).sum()
        print(f"DONE {inst}: {dates[0]} to {dates[-1]}, {len(dates)} days, {samples} samples")
        return samples

    except Exception as e:
        print(f"=== ERROR {e}", traceback.format_exc())
        # while True:
        #     pass
        # embed()
        return None

def gen_task(inst, dates):
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


with Pool(POOL_NUM) as p:
    tasks = p.starmap(gen_task, inst_dates.items())
tasks = [x for y in tasks for x in y]


def arange_task(tasks):
    tasks = sorted(tasks, key=lambda x: len(x[1]), reverse=True)
    task_list = [[] for _ in range(POOL_NUM)]
    task_list_len = np.zeros(POOL_NUM)
    for task in tasks:
        idx = task_list_len.argmin()
        task_list[idx].append(task)
        task_list_len[idx] += len(task[1])
    return task_list

tasks = arange_task(tasks)
for i in range(POOL_NUM):
    print(f"pool {i}: {len(tasks[i])} tasks, {sum([len(x[1]) for x in tasks[i]])} days")
# exit(0)


def do_gen_data(tasks, para=False):
    ret = 0
    for task in tasks:
        gen_data(task, para)
        # ret += gen_data(task, para)
    return ret

# do_gen_data(tasks[2])  # for test
# embed()
if not args.label:
    print("del data?")
    import time

    time.sleep(10)
    os.system(f"rm -rf {DEEP_DATA_DIR}")
# for t in tasks:
#     do_gen_data(t)
if EXCHANGE == "coinbase":
    ret = []
    for task in tasks:
        ret.append(do_gen_data(task, para=True))
else:
    with Pool(POOL_NUM) as p:
        ret = p.map(do_gen_data, tasks)

sample_num = sum([x for x in ret if x is not None])
print(f"total sample num: {sample_num / 1e4:.2f}w")