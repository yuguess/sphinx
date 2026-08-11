import os
import torch
import time
import json5
import pandas as pd
import pathlib
import traceback
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple, Optional
from multiprocessing import Pool

from mona.common.logging import get_logger
from mona.common import metadata
from sphinx.util.exchange_api import get_env_exchange
from core.model import gen_model
from .run_helper import parse_args, get_mdl_num, add_model_legacy_path, sanity_check, dump_log, gen_minute_encode_pred, decode_pred, get_cn_rnd_up_min_ts
from .cf_5min_sdk import create_infra_sdk, SDKWrapper


LOGGER = get_logger('cf_5min_run')


def do_min_infer(task_tm_s, cfg, univ, model_name, model_idx, sdk) -> Tuple[List[pd.Series], List[pd.Series]]:
        
    torch.set_num_threads(1)
    horizon = cfg[model_name]["horizon"]
    channel = cfg[model_name]["channel"]
    use_vola = cfg[model_name]["use_vola"]
    assert channel == len(horizon)
    checkpoints = sorted(os.listdir(cfg[model_name]["path"]))
    checkpoint = torch.load(os.path.join(cfg[model_name]["path"], checkpoints[model_idx], f"{cfg[model_name]['epoch']}.pth.tar"), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    all_feature_name_list = config["dataset"]["alphas"]
    assert all_feature_name_list[0] == "ret1m"
    
    # alpha_name_list = config["dataset"]["alphas"]
    model = gen_model(**config["model"], seq_len=config["dataset"]["seq_len"])
    model.load_state_dict(checkpoint[cfg[model_name]["state_dict_key"]])
    normalizer = checkpoint["normalizer"]
    
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
    
    market_list = [
        # f"market_universe_t60r1_oi_list30d_okx_futures",
        f"market_{cfg['universe']}",
        # "market_T30R20_r20to",
        # "market_T30R20_minuteto",
        # "market_T30R20_r20logto",
        # "market_T30R20_minutelogto",
    ]
    
    market_feature_name = [
        # "ret1m",
        "imbhl_r10",
        "imbhl_r40",
        "imbhl_r80",
        "imbhl_r160",
    ]
    
    inst_feature_name += [
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
    
    # list of df
    history_inst_self_feature_list = []
    for inst in univ:
        f_l = [sdk.deploy_read_history_alpha(task_tm_s, inst, f).rename(f) for f in inst_feature_name]
        history_inst_self_feature_list.append(pd.concat(f_l, axis=1))
        
    history_inst_feature_list = []        
    for inst_feature in history_inst_self_feature_list:
        history_inst_feature_list.append(pd.concat([inst_feature], axis=1)[all_feature_name_list].values.T)
        
    history_inst_feature = np.array(history_inst_feature_list)  
    # B = 20, C = 18, T = 1439
    history_inst_ret1m = history_inst_feature[:, 0:1, :]
    # 第一个 feature 必须是 ret1m！
    history_inst_ret1m = history_inst_ret1m[..., -(model.seq_len - 1) :]
    
    norm_history_inst_feature = (history_inst_feature / normalizer["feature"].std[None, :, None]).clip(-normalizer["feature"].clip, normalizer["feature"].clip)
    norm_history_inst_feature = norm_history_inst_feature[..., -(model.seq_len - 1) :]
    
    valid = pd.concat([sdk.deploy_read_history_alpha(task_tm_s, inst, "valid").eq(1).rename("valid") for inst in univ], axis=1).values
    
    # 设置状态, 训练集 gen_data 里有 max(10, h), 这里做检查
    assert all([h >= 10 for h in horizon])
    
    ewm_halflife = [h * 2 for h in horizon]
    ewm_alpha = [1 - np.exp(-np.log(2) / h) for h in ewm_halflife]
    ewm_mean = [np.zeros(len(universe)) for _ in horizon]
    ewm_var = [np.zeros(len(universe)) for _ in horizon]
    
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
                encode_pred[:, c], history_inst_ret1m[..., i:i + 1], ewm_alpha[c], ewm_mean[c], ewm_var[c], normalizer["label"].std[c:c + 1],
                valid[i], use_vola)
            preds[c].append(pred)
            prob = encode_prob[:, c] * 2 - 1
            probs[c].append(prob)
    # hist_df = pd.DataFrame(preds, index=history_index[-(model.seq_len - 1):], columns=universe)
    
    # 读最新数据
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "model read last alpha begin")
    read_last_st = time.time()
    last_inst_self_feature_list = []
    for inst in universe:
        df_l = [sdk.deploy_read_last_alpha(task_tm_s, inst, name).rename(name) for name in inst_feature_name]
        last_inst_self_feature_list.append(pd.concat(df_l, axis=1))

    last_inst_feature = []
    for inst_feature in last_inst_self_feature_list:
        last_inst_feature.append(pd.concat([inst_feature], axis=1)[all_feature_name_list].values.T)
        
    # B = 20, C = 18, T = 1
    last_inst_feature = np.array(last_inst_feature)  
      # 第一个 feature 必须是 ret1m！
    last_inst_ret1m = last_inst_feature[:, 0:1, :]
    norm_last_inst_feature = (last_inst_feature / normalizer["feature"].std[None, :, None]).clip(-normalizer["feature"].clip, normalizer["feature"].clip)
    last_valid = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "valid").eq(1).values[0] for inst in universe])
    
    last_encode_pred, last_encode_prob = gen_minute_encode_pred(model, norm_last_inst_feature)
    last_preds = []
    last_probs = []
    for c in range(channel):
        last_pred, last_ewm_mean, last_ewm_var = decode_pred(
            last_encode_pred[:, c], last_inst_ret1m, ewm_alpha[c], ewm_mean[c], ewm_var[c], normalizer["label"].std[c:c + 1], last_valid, use_vola)
        last_preds.append(last_pred)
        last_prob = last_encode_prob[:, c] * 2 - 1
        last_probs.append(last_prob)

    last_preds = [pd.Series(p, index=universe) for p in last_preds]
    last_probs = [pd.Series(p, index=universe) for p in last_probs]

    LOGGER.info(f"[{model_id}] read last cost {(time.time() - read_last_st)*1e3:.2f}ms")
    return last_preds, last_probs
    

def do_min_infer_wrapper(task_tm_s, cfg, univ, model_name, model_idx, trade_dt_s) -> Optional[Tuple[List[pd.Series], List[pd.Series]]]:
    engles_cli = create_infra_sdk(trade_dt_s, cfg)
    try:
        return do_min_infer(task_tm_s, cfg, univ, model_name, model_idx, engles_cli)
    except Exception as e:
        dump_log(f"[{model_name} {model_idx} ERROR], {e}", traceback.format_exc())
        return None


class LoopCtx:

    def __init__(self, cfg, parent_cfg_path, hist_row_csv_path, sdk_cli: SDKWrapper):
        self.account_name = cfg["name"]
        self.strategy_cfg = cfg["strategy"]
        self.stage = len(cfg["strategy"]["max_beta_exposure"])

        if hist_row_csv_path is None:
            self.hist_row = [pd.Series(np.nan) for _ in range(self.stage)]
        elif hist_row_csv_path == "deploy":
            self.hist_row = [sdk_cli.deploy_read_last_holding(self.account_name) / self.stage for _ in range(self.stage)]
        else:
            df = pd.read_csv(hist_row_csv_path, index_col=0)
            self.hist_row = [df.loc[f"holding_stage{i}(%)"] / 100 for i in range(self.stage)]
            assert f"holding_stage{self.stage}(%)" not in df.index

        self.deploy_last_row = pd.Series(np.nan)
        self.fusion_row = [pd.Series(np.nan) for _ in range(self.stage)]
        self.nav = -1
        self.run_name = ""
        self.parent_cfg_path = parent_cfg_path
        self.curr_date = ""
        self.valid_insts = pd.Series()

    def reset_holding(self):
        self.hist_row = [pd.Series(np.nan) for _ in range(self.stage)]
        self.deploy_last_row = pd.Series(np.nan)
        self.fusion_row = [pd.Series(np.nan) for _ in range(self.stage)]

    def try_reset(self, xhg, task_date, sdk_cli):
        curr_nav = sdk_cli.deploy_read_nav(self.account_name)

        if curr_nav != self.nav or self.curr_date != task_date:
            # if curr_nav != self.nav:
            #     self.reset_holding()
            self.curr_date = task_date
            self.nav = curr_nav
            self.run_name = f"{self.account_name}/{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
            os.makedirs(f"deploy/data/{self.run_name}", exist_ok=True)
            os.system(
                f'cp "{self.parent_cfg_path}" "deploy/data/{self.run_name}/config.json5"'
            )
            os.system(f'echo "{curr_nav}" > "deploy/data/{self.run_name}/nav.txt"')
            LOGGER.info(f"run_name: {self.run_name}")
        

def pred_run(cfg, xhg, model_num, task_time_s, trade_dt_s, loop_ctx: LoopCtx):
    infra_sdk = create_infra_sdk(trade_dt_s, cfg)
    loop_ctx.try_reset(xhg, trade_dt_s, infra_sdk)
        
    if not infra_sdk.is_trading_time(task_time_s):
        return loop_ctx
    
    univ = infra_sdk.read_universe(cfg["universe"])
    valid_univ = pd.Series(1, index=univ)
    
    task_arg_tps = [(task_time_s, cfg, univ, 'model', mdl_idx, trade_dt_s) for mdl_idx in range(model_num)]
    with Pool(model_num) as p:
        ret = p.starmap(do_min_infer_wrapper, task_arg_tps)
    LOGGER.info("model infer done")
    # ret shape is like  #Model X 2 X #Channel
    fusion_preds = [sum([r[0][i] for r in ret]) / model_num for i in range(len(ret[0][0]))]
    fusion_prob_preds = [sum([r[1][i] for r in ret]) / model_num for i in range(len(ret[0][1]))]
    
    last_turnover = None
    last_bid1_price = None
    last_ask1_price = None
    last_bid1_volume = None
    last_ask1_volume = None
    
    fee = infra_sdk.deploy_read_fee_rates(loop_ctx.account_name).reindex(univ).fillna(0).values
    last_inst_close_price = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "close_price").values[0] for inst in univ])

    last_std = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, cfg["std_name"]).values[0] for inst in univ])
    last_valid = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "valid").eq(1).values[0] for inst in univ])
    
    turnover_ma0 = None
    turnover_ma1 = None
    turnover_ma2 = None
    
    # corr1 = None
    # corr2 = None
    # corr3 = None
    # oi = None
    # funding_fee = None
    
    if loop_ctx.strategy_cfg["exec_info"]["exec_type"] in ["make", "make2"] and last_turnover is None:
        last_turnover = np.array([infra_sdk.deploy_read_last_turnover(task_time_s, inst).values[0] for inst in univ])
        last_turnover[~np.isfinite(last_turnover)] = 0
        half_spread_mean = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "half_spread_mean").values[0] for inst in univ])
        book1_value_sum0 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "book1_value_sum_1s_r1").values[0] for inst in univ])
        book1_value_sum1 = np.array([infra_sdk.deploy_read_last_alpha(task_time_s, inst, "book1_value_sum_1s_r200").values[0] for inst in univ])
    elif loop_ctx.strategy_cfg["exec_info"]["exec_type"] == "take":
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
    
    loop_ctx.deploy_last_row = deploy_read_last_holding(engles_cli, loop_ctx.account_name).reindex(univ).fillna(0)
    loop_ctx.hist_row = [h.reindex(univ).fillna(0) for h in loop_ctx.hist_row]
    
    loop_ctx.valid_insts = valid_univ.copy()
    for l in loop_ctx.hist_row:
        loop_ctx.valid_insts[l[l.abs() > 1e-6].index] = 1
    
    fusion_row = do_minute_opt(
        cfg=loop_ctx.strategy_cfg,
        last_row=loop_ctx.hist_row,
        signals=fusion_preds,
        probs=fusion_prob_preds,
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
        turnover_ma2=turnover_ma2,
        ts=task_time_s,
        fee=fee)
    
    fusion_row_df = pd.DataFrame(
        [sum(fusion_row).values, np.zeros_like(fusion_row[0]), last_inst_close_price], index=["holding", "ty", "close_price"],
        columns=fusion_row[0].index).T

    deploy_write_holding(engles_cli, task_time_s, fusion_row_df, loop_ctx.strategy_cfg["ty"])
    
    loop_ctx.hist_row = fusion_row
    
    # audit booking 
    last_ret1ms = pd.Series([engles_cli.deploy_read_last_alpha(task_time_s, inst, "ret1m").values[0] for inst in univ], index=univ).mul(1e4).rename("ret1m(bp)")
    last_midret1ms = pd.Series([engles_cli.deploy_read_last_alpha(task_time_s, inst, "midret1m").values[0] for inst in univ], index=univ).mul(1e4).rename("midret1m(bp)")
    vwap_slippage = pd.Series([engles_cli.deploy_read_last_alpha(task_time_s, inst, "vwap_slippage").values[0] for inst in univ], index=univ).mul(1e4).rename("vwap_slippage(bp)")
    preds = [p.mul(1e4).rename(f"pred{i + 1}(bp)") for i, p in enumerate(fusion_preds)]
    prob_preds = [p.rename(f"prob_pred{i + 1}") for i, p in enumerate(fusion_prob_preds)]
    
    info = pd.concat([
        sum(fusion_row).rename("holding(%)").mul(100).to_frame(),
        *[p.to_frame() for p in preds],
        *[p.to_frame() for p in prob_preds],
        last_ret1ms.to_frame(),
        last_midret1ms.to_frame(),
        vwap_slippage.to_frame(),
        # vwap_slippage_mid.to_frame(),
        ],axis=1).T
    
    info.loc["limit_make"] = False
    info.loc["close_price"] = last_inst_close_price
    info.loc["real_holding(%)"] = loop_ctx.deploy_last_row.mul(100)
    info.loc["valid"] = last_valid & loop_ctx.valid_insts.eq(1)

    info.loc["fee(bp)"] = fee * 1e4
    if loop_ctx.strategy_cfg["exec_info"]["exec_type"] == "make":
        info.loc["last_turnover(1e6)"] = last_turnover / 1e6
    elif loop_ctx.strategy_cfg["exec_info"]["exec_type"] == "take":
        info.loc["last_bid1_price"] = last_bid1_price
        info.loc["last_ask1_price"] = last_ask1_price
        info.loc["last_bid1_volume"] = last_bid1_volume
        info.loc["last_ask1_volume"] = last_ask1_volume
        info.loc["last_bid1_turnover"] = last_bid1_price * last_bid1_volume
        info.loc["last_ask1_turnover"] = last_ask1_price * last_ask1_volume
        vwap_slippage_mid = (
            pd.Series([engles_cli.deploy_read_last_alpha(task_time_str, inst, "vwap_slippage_mid").values[0] for inst in universe], index=universe).mul(1e4).rename("vwap_slippage_mid(bp)"))
        info.loc["vwap_slippage_mid(bp)"] = vwap_slippage_mid
        
    info.loc["last_var(1e-5)"] = last_std * last_std * 1e5
    # exchange_api = create_exchange_api(account.account_name)
    # real_qty_holding = pd.Series(exchange_api.query_position())
    # info.loc["real_qty_holding"] = real_qty_holding.reindex(info.columns).fillna(0)
    # info.loc["real_equity"] = exchange_api.query_account_information().margin_balance
    info.loc["real_equity"] = deploy_read_equity(engles_cli, loop_ctx.account_name)
    for i, h in enumerate(loop_ctx.hist_row):
        info.loc[f"holding_stage{i}(%)"] = h.mul(100)
    info = info.round(3)
    tmp = info.copy()
    LOGGER.info(tmp.round(3))
    LOGGER.info(f"holding sum: {info.loc['real_holding(%)'].sum():.3f}%")
    LOGGER.info(f"holding abs sum: {info.loc['real_holding(%)'].abs().sum():.3f}%")
    info.to_csv(f"deploy/data/{loop_ctx.run_name}/{task_time_s}.csv")
    LOGGER.info(f"task {task_time_s} done")

    return loop_ctx


def get_trade_date(xhg, time_str: str) -> Optional[str]:
    ts = pd.Timestamp(time_str, tz="Asia/Shanghai")
    date = ts.date().strftime("%Y-%m-%d")
    if xhg == "CF" and ts.weekday() >= 5:
        # if weekends, return last friday
        dates = metadata.dates((ts - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), date)
        if len(dates) == 0:
            return None
        date = dates[-1]
    try:
        metadata.index(date)
    except ValueError:
        return None
    def get_date_info(d: str):
        index = metadata.index(d)
        start = index[0] - INDEX_INTERVAL
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

def main():
    xhg = get_env_exchange()
    
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json5.load(f)
        
    add_model_legacy_path(str(pathlib.Path(__file__).parent))
    model_num = get_mdl_num(cfg)
    
    sanity_check(cfg, xhg, model_num)

    loop_ctx = 

    while True:        
        task_min_s = get_cn_rnd_up_min_ts()
        opt_trade_date = get_trade_date(xhg, task_min_s)

        if ts.second <= 35 or opt_trade_date is None:
            time.sleep(1)
        else:
            loop_ctx = pred_run(cfg, xhg, model_num, task_min_s, opt_trade_date, loop_ctx)
            

if __name__ == '__main__':
    main()
