import time
import os
import pathlib
import traceback
from typing import List, Tuple, Optional
from multiprocessing import Pool
import torch
import psutil
import numpy as np
import pandas as pd
import json5

from mona.common import INDEX_INTERVAL
from mona.common import metadata
from mona.common.logging import get_logger
from sphinx.util.exchange_api import get_env_exchange
from ..run_helper import parse_args, get_mdl_num, add_model_legacy_path, sanity_check, get_cn_rnd_up_min_ts, dump_log, gen_minute_encode_pred, decode_pred
from ..run_helper import print_open_file, v0_min_infer
from ..okx_5min_sdk import create_infra_sdk, SDKWrapper, is_trading_time
from ..run_adt import LoopCtx
# from .bn_5min_opt import GenPortfolio


LOGGER = get_logger('okx_10min')


def get_trade_date(time_str: str, xhg, idx_interval) -> Optional[str]:
    ts = pd.Timestamp(time_str, tz='Asia/Shanghai')
    date = ts.date().strftime('%Y-%m-%d')
    if xhg in ['CF', "CF5m"] and ts.weekday() >= 5:
        # if weekends, return last friday
        dates = metadata.dates((ts - pd.Timedelta(days=3)).strftime('%Y-%m-%d'), date)
        if len(dates) == 0:
            return None
        date = dates[-1]
    try:
        metadata.index(date)
    except ValueError:
        return None

    def get_date_info(d: str):
        index = metadata.index(d)
        start = index[0] - idx_interval
        end = index[-1]
        return start, end

    side = None
    cnt = 0
    while True:
        start, end = get_date_info(date)
        if cnt > 1000:
            raise RuntimeError(f"too many iterations: {cnt}")
        if start < ts <= end:
            return date
        if ts <= start:
            if side == 'next':
                return None
            date = metadata.prev_date(date)
            side = 'prev'
        if ts > end:
            if side == 'prev':
                return None
            date = metadata.next_date(date)
            side = 'next'
        cnt += 1


def do_min_infer_wrapper(task_tm_s, cfg, univ, model_name, model_idx, trade_dt_s) -> Optional[Tuple[List[pd.Series], List[pd.Series]]]:
    sdk = create_infra_sdk(trade_dt_s, cfg)
    try:
        return v0_min_infer(task_tm_s, cfg, univ, model_name, model_idx, sdk)
    except Exception as e:
        LOGGER.error(f"[{model_name} {model_idx} ERROR], {traceback.format_exc()}")
        dump_log(f"[{model_name} {model_idx} ERROR], {e}", traceback.format_exc())
        return None
    
    
def pred_run(cfg, xhg, model_num, task_time_s, trade_dt_s, loop_ctx: LoopCtx, infra_sdk: SDKWrapper) -> LoopCtx:
    print_open_file()

    loop_ctx.try_reset(xhg, trade_dt_s, infra_sdk)

    LOGGER.info("run trade date:%s, task_min:%s", trade_dt_s, task_time_s)
    if not infra_sdk.is_trading_time(task_time_s):
        return loop_ctx

    univ = infra_sdk.deploy_read_universe(cfg["universe"])
    univ_2d = infra_sdk.deploy_read_universe(f'{cfg["universe"]}_2d')
    
    valid_univ = pd.Series(1, index=univ)
    
    task_arg_tps = [(task_time_s, cfg, univ, 'model', mdl_idx, trade_dt_s) for mdl_idx in range(model_num)]
    with Pool(model_num) as p:
        ret = p.starmap(do_min_infer_wrapper, task_arg_tps)
    LOGGER.info("model infer done")
    # ret shape is like  #Model X 2 X #Channel
    fusion_preds = [sum([r[0][i] for r in ret]) / model_num for i in range(len(ret[0][0]))]
    fusion_prob_preds = [sum([r[1][i] for r in ret]) / model_num for i in range(len(ret[0][1]))]
    

    # prepare data for opt
    last_turnover = None
    last_bid1_price = None
    last_ask1_price = None
    last_bid1_volume = None
    last_ask1_volume = None
    half_spread_mean = None
    book1_value_sum0 = None
    book1_value_sum1 = None

    fee = pd.Series(cfg["account"][0]["strategy"]["fee"], index=univ).values

    last_inst_close_price = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "close_price").values[0] for inst in univ])
    # last_inst_bid1_price = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "bid1_price").values[0] for inst in universe])
    # last_inst_ask1_price = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "ask1_price").values[0] for inst in universe])
    last_std = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, cfg["std_name"]).values[0] for inst in univ])
    last_valid = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "valid").eq(1).values[0] for inst in univ])
    ret1m = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "ret1m").values[0] for inst in univ])
    
    turnover_ma0 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "turnover_r1").values[0] for inst in univ])
    turnover_ma1 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "turnover_r200").values[0] for inst in univ])
    # turnover_ma2 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "turnover_r200").values[0] for inst in universe])
    corr1 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, f"corr_market_{cfg['universe']}_r10").values[0] for inst in univ])
    corr2 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, f"corr_market_{cfg['universe']}_r40").values[0] for inst in univ])
    corr3 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, f"corr_market_{cfg['universe']}_r160").values[0] for inst in univ])
    oi = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "open_interest").values[0] for inst in univ])
    funding_fee = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "funding_1hr_r24h").values[0] for inst in univ])
    
    if loop_ctx.strategy_cfg["exec_info"]["exec_type"] in ["make", "make2"] and last_turnover is None:
        last_turnover = np.array([infra_sdk.deploy_read_last_turnover(task_time_s, inst).values[0] for inst in univ])
        last_turnover[~np.isfinite(last_turnover)] = 0
        half_spread_mean = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "half_spread_mean").values[0] for inst in univ])
        book1_value_sum0 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "book1_value_sum_1s_r1").values[0] for inst in univ])
        book1_value_sum1 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "book1_value_sum_1s_r200").values[0] for inst in univ])
    elif loop_ctx.strategy_cfg["exec_info"]["exec_type"] == "take" and last_bid1_price is None and last_ask1_price is None and last_bid1_volume is None and last_ask1_volume is None:
        last_bid1_price = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "bid1_price").values[0] for inst in univ])
        last_ask1_price = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "ask1_price").values[0] for inst in univ])
        last_bid1_volume = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "bid1_volume").values[0] for inst in univ])
        last_ask1_volume = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "ask1_volume").values[0] for inst in univ])
        last_bid1_price[~np.isfinite(last_bid1_price)] = 0
        last_ask1_price[~np.isfinite(last_ask1_price)] = 0
        last_bid1_volume[~np.isfinite(last_bid1_volume)] = 0
        last_ask1_volume[~np.isfinite(last_ask1_volume)] = 0
    else:
        raise ValueError("unknown exec type")
    
    loop_ctx.deploy_last_row = infra_sdk.deploy_read_last_holding(loop_ctx.account_name).reindex(univ).fillna(0)
    loop_ctx.hist_row = [h.reindex(univ).fillna(0) for h in loop_ctx.hist_row]
    loop_ctx.valid_insts = valid_univ.copy()
    for l in loop_ctx.hist_row:
        loop_ctx.valid_insts[l[l.abs() > 1e-6].index] = 1
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "opt begin")

    fusion_row = do_minute_opt(
        cfg=loop_ctx.strategy_cfg,
        last_row=loop_ctx.hist_row,
        # signals=fusion_preds,
        # probs=fusion_prob_preds,
        signals=fusion_preds[:1],
        probs=fusion_prob_preds[:1],
        last_price=last_inst_close_price,
        last_std=last_std,
        last_turnover=last_turnover,
        last_bid1_price=last_bid1_price,
        last_ask1_price=last_ask1_price,
        last_bid1_volume=last_bid1_volume,
        last_ask1_volume=last_ask1_volume,
        last_valid=last_valid & loop_ctx.valid_insts.eq(1).values,
        turnover_ma0=turnover_ma0,
        turnover_ma1=turnover_ma1,
        # turnover_ma2=turnover_ma2,
        corr1=corr1,
        corr2=corr2,
        corr3=corr3,
        oi=oi,
        ret1m=ret1m,
        ts=task_time_s,
        fee=fee,
        funding_fee=funding_fee,
        half_spread_mean=half_spread_mean,
        book1_value_sum0=book1_value_sum0,
        book1_value_sum1=book1_value_sum1,
    )

    fusion_row_df = pd.DataFrame(
        [sum(fusion_row).values, np.zeros_like(fusion_row[0]), last_inst_close_price],
        index=["holding", "ty", "close_price"], columns=fusion_row[0].index).T
    LOGGER.info(f"debug fusion_row:{task_time_s},\n{fusion_row_df}")

    


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json5.load(f)

    model_num = get_mdl_num(cfg)
    xhg = get_env_exchange()
    sanity_check(cfg, xhg, model_num)

    task_min_s = get_cn_rnd_up_min_ts()
    trd_date_s = get_trade_date(task_min_s, xhg, INDEX_INTERVAL)
    sdk_cli = create_infra_sdk(trd_date_s, cfg)
    loop_ctx = LoopCtx(cfg["account"][0], args.config, args.prev_data_csv_path, sdk_cli)
    
    while True:
        ts = pd.Timestamp.now(tz="Asia/Shanghai")
        task_min_ts = ts.floor("min") + pd.Timedelta(minutes=1)
        task_min_s = task_min_ts.strftime("%Y-%m-%d %H:%M:%S")
        opt_trade_date = get_trade_date(task_min_s, xhg, INDEX_INTERVAL)

        if opt_trade_date is None or not is_trading_time(opt_trade_date, task_min_s):
            time.sleep(1)
        else:
            infra_sdk = create_infra_sdk(opt_trade_date, cfg)
            try:
                loop_ctx = pred_run(cfg, xhg, model_num, task_min_s, opt_trade_date, loop_ctx, infra_sdk)
            except Exception as e:
                LOGGER.error(f"OPT ERROR:{e}, {traceback.format_exc()}")
                dump_log(f"[OPT ERROR]", traceback.format_exc())
            finally:
                infra_sdk.close()


if __name__ == '__main__':
    add_model_legacy_path(str(pathlib.Path(__file__).parent))
    main()