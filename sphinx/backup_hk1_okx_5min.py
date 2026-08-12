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
from sphinx.core.model import gen_model
from .run_helper import parse_args, get_mdl_num, add_model_legacy_path, sanity_check, get_cn_rnd_up_min_ts, dump_log, gen_minute_encode_pred, decode_pred
from .run_helper import print_open_file, v0_min_infer
from .okx_5min_sdk import create_infra_sdk, SDKWrapper, is_trading_time
from .okx_5min_opt import GenPortfolio
from .run_adt import LoopCtx


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


def do_minute_opt(
    cfg,
    last_row,
    signals,
    probs,
    last_price,
    last_std,
    last_turnover,
    last_bid1_price,
    last_ask1_price,
    last_bid1_volume,
    last_ask1_volume,
    last_valid,
    turnover_ma0,
    turnover_ma1,
    # turnover_ma2,
    corr1,
    corr2,
    corr3,
    oi,
    funding_fee,
    half_spread_mean,
    book1_value_sum0,
    book1_value_sum1,
    ret1m,
    ts,
    fee,
):
    org_last_row = [r.copy() for r in last_row]
    try:
        universe = last_row[0].index
        # FEE = cfg["fee"]

        # TODO：opt 全局生成一次就够了，不应该每次都生成，浪费时间
        opt = GenPortfolio(
            alpha_name=["" for _ in range(len(signals))],
            prob_name=["" for _ in range(len(signals))],
            univ_name="",
            std_name="",  # 不该被使用
            nav=cfg["nav"],
            signal_coef=cfg["signal_coef"],
            # open_prob_thres=cfg["open_prob_thres"],
            # close_prob_thres=cfg["close_prob_thres"],
            inst_risk_coef=cfg["inst_risk_coef"],
            exec_info=cfg["exec_info"],
            max_inst_exposure=cfg["max_inst_exposure"],
            max_beta_exposure=cfg["max_beta_exposure"],
            open_cost_coef=cfg["open_cost_coef"],
            close_cost_coef=cfg["close_cost_coef"],
            max_open_turnover=cfg["max_open_turnover"],
            max_close_turnover=cfg["max_close_turnover"],
            # abnormal_turnver_ban_thres1=cfg["abnormal_turnver_ban_thres1"],
            # abnormal_turnver_ban_thres2=cfg["abnormal_turnver_ban_thres2"],
            abnormal_turnver_ban_open_beta=cfg["abnormal_turnver_ban_open_beta"],
            abnormal_corr_ban_open_thres=cfg["abnormal_corr_ban_open_thres"],
            signal_horizon=cfg["signal_horizon"],
            req_margin=cfg["req_margin"],
            funding_fee_coef=cfg["funding_fee_coef"],
        )
        opt_task = opt.build_task(universe, None)

        # TODO：为 take mode 生成对应的变量
        if cfg["exec_info"]["exec_type"] == "make":
            assert opt.exec_topk == 1
            adjust_book1_value_sum0 = book1_value_sum0 * opt.book_limit_rate / opt.turnover_limit_rate
            short_price = [last_price * (1 - opt.extra_slippage[k]) for k in range(opt.exec_topk)]
            # short_vol = [last_turnover.where(last_turnover < adjust_book1_value_sum0, adjust_book1_value_sum0) / short_price[k] / opt.exec_topk for k in range(opt.exec_topk)]
            short_vol = [np.where(last_turnover < adjust_book1_value_sum0, last_turnover, adjust_book1_value_sum0) / short_price[k] / opt.exec_topk for k in range(opt.exec_topk)]
            long_price = [last_price * (1 + opt.extra_slippage[k]) for k in range(opt.exec_topk)]
            long_vol = [np.where(last_turnover < adjust_book1_value_sum0, last_turnover, adjust_book1_value_sum0) / long_price[k] / opt.exec_topk for k in range(opt.exec_topk)]
        elif cfg["exec_info"]["exec_type"] == "make2":
            assert opt.exec_topk == 1
            adjust_book1_value_sum0 = book1_value_sum0 * opt.book_limit_rate / opt.turnover_limit_rate
            short_price = [last_price - half_spread_mean for k in range(opt.exec_topk)]
            short_vol = [np.where(last_turnover < adjust_book1_value_sum0, last_turnover, adjust_book1_value_sum0) / short_price[k] / opt.exec_topk for k in range(opt.exec_topk)]
            long_price = [last_price + half_spread_mean for k in range(opt.exec_topk)]
            long_vol = [np.where(last_turnover < adjust_book1_value_sum0, last_turnover, adjust_book1_value_sum0) / long_price[k] / opt.exec_topk for k in range(opt.exec_topk)]
        elif cfg["exec_info"]["exec_type"] == "take":
            assert opt.exec_topk == 1
            short_price = [last_bid1_price]
            short_vol = [last_bid1_volume]
            long_price = [last_ask1_price]
            long_vol = [last_ask1_volume]
        else:
            raise ValueError("unknown exec type")

        # print(fee)
        last_row = opt.update_one_line(
            task=opt_task,
            last_row=last_row,
            alphas=[s.reindex(universe).fillna(0).values for s in signals],
            # probs=[p.reindex(universe).fillna(0).values for p in probs],
            std=last_std,
            valid=last_valid,
            fee=fee,
            short_price=short_price,
            short_vol=short_vol,
            long_price=long_price,
            long_vol=long_vol,
            turnover_ma0=turnover_ma0,
            turnover_ma1=turnover_ma1,
            # turnover_ma2=turnover_ma2,
            corr1=corr1,
            corr2=corr2,
            corr3=corr3,
            oi=oi,
            funding_fee=funding_fee,
            ret1m=ret1m,
            book1_value_sum0=book1_value_sum0,
            book1_value_sum1=book1_value_sum1,
            timestamp=ts,
        )
        return last_row

    except Exception as e:
        LOGGER.error("OPT ERROR:%s, %s", str(e), str(traceback.format_exc()))
        dump_log("[OPT ERROR] %s, %s", str(e), str(traceback.format_exc()))
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
    LOGGER.info(f"opt fusion_row:{task_time_s},\n{fusion_row_df}")

    # infra_sdk.deploy_write_holding(task_time_s, fusion_row_df, loop_ctx.strategy_cfg["ty"])

    # # write close holding
    # close_symbols = sorted(set(univ_2d) - set(univ))
    # if len(close_symbols) > 0:
    #     close_holding = pd.Series(0.0, index=close_symbols).to_frame(name='holding')
    #     infra_sdk.deploy_write_holding(task_time_s, close_holding, infra_sdk.strategy_cfg["ty"])

    loop_ctx.hist_row = fusion_row

    # audit booking 
    last_ret1ms = pd.Series([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "ret1m").values[0] for inst in univ], index=univ).mul(1e4).rename("ret1m(bp)")
    last_midret1ms = pd.Series([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "midret1m").values[0] for inst in univ], index=univ).mul(1e4).rename("midret1m(bp)")
    vwap_slippage = pd.Series([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "vwap_slippage").values[0] for inst in univ], index=univ).mul(1e4).rename("vwap_slippage(bp)")
    preds = [p.mul(1e4).rename(f"pred{i + 1}(bp)") for i, p in enumerate(fusion_preds)]
    prob_preds = [p.rename(f"prob_pred{i + 1}") for i, p in enumerate(fusion_prob_preds)]

    info = pd.concat(
        [
            sum(fusion_row).rename("holding(%)").mul(100).to_frame(),
            *[p.to_frame() for p in preds],
            *[p.to_frame() for p in prob_preds],
            last_ret1ms.to_frame(),
            last_midret1ms.to_frame(),
            vwap_slippage.to_frame(),
            # vwap_slippage_mid.to_frame(),
        ], axis=1).T

    info.loc["limit_make"] = False
    info.loc["close_price"] = last_inst_close_price
    info.loc["real_holding(%)"] = loop_ctx.deploy_last_row.mul(100)
    info.loc["valid"] = last_valid & loop_ctx.valid_insts.eq(1)

    info.loc["turnover_ma0(1e6)"] = turnover_ma0 / 1e6
    info.loc["turnover_ma1(1e6)"] = turnover_ma1 / 1e6
    # info.loc["turnover_ma2(1e6)"] = turnover_ma2 / 1e6
    info.loc["turnover_abnormal_coef1"] = turnover_ma0.sum() / turnover_ma1.sum()
    # info.loc["turnover_abnormal_coef2"] = turnover_ma1 / turnover_ma2
    info.loc["corr1"] = corr1
    info.loc["corr2"] = corr2
    info.loc["corr3"] = corr3
    info.loc["oi(1e6)"] = oi / 1e6
    info.loc["funding_fee(bp)"] = funding_fee * 1e4
    info.loc["fee(bp)"] = fee * 1e4

    if loop_ctx.strategy_cfg["exec_info"]["exec_type"] in ["make", "make2"]:
        info.loc["last_turnover(1e6)"] = last_turnover / 1e6
        info.loc["half_spread_mean/close(bp)"] = half_spread_mean / last_inst_close_price * 1e4
    elif loop_ctx.strategy_cfg["exec_info"]["exec_type"] == "take":
        info.loc["last_bid1_price"] = last_bid1_price
        info.loc["last_ask1_price"] = last_ask1_price
        info.loc["last_bid1_volume"] = last_bid1_volume
        info.loc["last_ask1_volume"] = last_ask1_volume
        info.loc["last_bid1_turnover"] = last_bid1_price * last_bid1_volume
        info.loc["last_ask1_turnover"] = last_ask1_price * last_ask1_volume
        vwap_slippage_mid = pd.Series(
            [infra_sdk.deploy_read_last_alpha(task_time_s, inst, "vwap_slippage_mid").values[0] for inst in univ], index=univ).mul(1e4).rename("vwap_slippage_mid(bp)")
        info.loc["vwap_slippage_mid(bp)"] = vwap_slippage_mid
    else:
        raise ValueError("unknown exec type")

    info.loc["last_var(1e-5)"] = last_std * last_std * 1e5
    # exchange_api = create_exchange_api(account.account_name)
    # real_qty_holding = pd.Series(exchange_api.query_position())
    # info.loc["real_qty_holding"] = real_qty_holding.reindex(info.columns).fillna(0)
    # info.loc["real_equity"] = exchange_api.query_account_information().margin_balance
    info.loc["real_equity"] = infra_sdk.deploy_read_equity(loop_ctx.account_name)
    for i, h in enumerate(loop_ctx.hist_row):
        info.loc[f"holding_stage{i}(%)"] = h.mul(100)

    info = info.round(3)
    tmp = info.copy()
    tmp.columns = [i.split("-")[1] for i in tmp.columns]

    LOGGER.info("\n%s", str(tmp.round(3)))
    LOGGER.info(f"holding sum: {info.loc['real_holding(%)'].sum():.3f}%")
    LOGGER.info(f"holding abs sum: {info.loc['real_holding(%)'].abs().sum():.3f}%")
    info.to_csv(f"deploy/data/{loop_ctx.run_name}/{task_time_s}.csv")
    LOGGER.info("task %s done", task_time_s)

    return loop_ctx


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
