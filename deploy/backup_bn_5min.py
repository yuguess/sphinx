import os
import math
import json5
import time
import psutil
import traceback
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Optional

from mona.common import metadata, INDEX_INTERVAL, db
from mona.common.util import dbtool
from mona import sdk

from sphinx.core.model import gen_model
from sphinx.deprecated.main2 import GenPortfolio


EXCHANGE = os.environ["EXCHANGE"]
assert EXCHANGE in ["CF5m", "okx10m", "binance5m", "okx5m"], f"Unknown exchange: {EXCHANGE}"

def dump_log(*args):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(f"error_{t}.txt", "w") as f:
        print(*args)
        print(*args, file=f)


def parse_args():
    parser = ArgumentParser(description='infer')
    parser.add_argument('config', type=str)
    parser.add_argument('-p', '--prev_data_csv_path', type=str, default=None)
    parser.add_argument('--is_test', action='store_true')
    return parser.parse_args()


def get_trade_date(time_str: str) -> Optional[str]:
    ts = pd.Timestamp(time_str, tz='Asia/Shanghai')
    date = ts.date().strftime('%Y-%m-%d')
    if EXCHANGE in ['CF', "CF5m"] and ts.weekday() >= 5:
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


def assert_mdl_conf(cfg) -> int:
    # assert "baseline" in cfg["model"]["path"]
    for h in cfg["model"]["horizon"]:
        assert f"{h}m" in cfg["model"]["path"]    

    model_num = len(os.listdir(cfg["model"]["path"]))

    # 不会自动从 deploy 里读最新仓位，以 0 开始
    # account_list = [Account(c, args.config, args.prev_data_csv_path) for c in cfg["account"]]
    account_list = None
    if EXCHANGE in ["okx10m", "binance5m", "okx5m", "CF5m"]:
        assert model_num == 8
    elif EXCHANGE == "CF":
        assert model_num == 4
    else:
        assert EXCHANGE is not None
        raise ValueError("Unknown exchange: " + EXCHANGE)

    return model_num


def check_print_file_handles():
    current_process = psutil.Process()
    file_handles = current_process.open_files()
    print(f"普通文件句柄数量: {len(file_handles)}")
    # 查看具体文件路径
    # for handle in file_handles:
    #     print(f"FD: {handle.fd}, Path: {handle.path}")


class SDKWrapper:

    def __init__(self, date: str, net_mode: bool, univ_name: str, accounts: list[str]) -> None:
        self._date = date
        self._net_mode = net_mode
        self._ctx = sdk.SHMContext(date)
        self._univ_name = univ_name
        if accounts is None:
            accounts = self._find_accounts()
        assert len(accounts) < 20
        if diff := set(accounts) - set(self._find_accounts()):
            raise ValueError(f"accounts {diff} not found")
        self._ectxs = {a: sdk.ExeSHMContext(date, a, net_mode=net_mode, write_safe=False) for a in accounts}
        self._date = date
        self._index = sdk.metadata.index(date)
        self._full_index = sdk.metadata.index_with_history(date, sdk.DEPLOY_HISTORY_LENGTH)
        self._full_index_ts = self._full_index.astype(int) / 1e9
        self._source_map = {
            'close_price': 'kline/close',
            'turnover': 'kline/turnover',
            'bid1_price': 'quotes/bid1_price',
            'ask1_price': 'quotes/ask1_price',
            'bid1_volume': 'quotes/bid1_volume',
            'ask1_volume': 'quotes/ask1_volume',
            'open_interest': 'oi/oi_quote',
        }

        days_num = math.ceil(sdk.DEPLOY_HISTORY_LENGTH / self._index.shape[0])
        dates = metadata.dates(metadata.calc_date(self._date, -days_num), self._date)
        if EXCHANGE in ['CF', "CF5m"]:
            raise NotImplementedError("CF exchange not supported for index groups")
            # self._index_groups = futures_metadata.index_groups(self._date)
            # self._group_index_ts = {g: pd.DatetimeIndex([i for date in dates for i in futures_metadata.index(date, security_group=g)]).astype(int) / 1e9 for g in self._index_groups}
            # self._symbols_group = futures_metadata.symbols_group(self._date)
        else:
            self._index_groups = None
            self._group_index_ts = None
            self._symbols_group = None

        relative_index = (self._index.shape[0] - np.arange(self._full_index.shape[0]) - 1)[::-1]
        if EXCHANGE in ['CF', "CF5m"]:
            raise NotImplementedError("CF exchange not supported")
            # exec_valid = pd.concat([futures_metadata.exec_valid(d).shift(-1).fillna(0).astype(bool) for d in dates]).iloc[-len(relative_index):]
            # self._time_valid = exec_valid.reindex(columns=self._symbols_group.index)
        else:
            self._time_valid = pd.concat([self._read_time_valid(i) for i in relative_index], axis=1).T

        self._time_valid.index = relative_index
        # apply night session mask if CF
        self._time_valid_normal = self._time_valid.copy()
        if EXCHANGE in ['CF', "CF5m"]:
            raise NotImplementedError("CF exchange not supported")
            # full_night_session_mask = pd.concat([self._night_session_mask(date) for date in dates], axis=0).reindex(self._full_index)
            # self._time_valid.loc[~full_night_session_mask.values] = False

        self.last_alpha_cache = {}

    def _night_session_mask(self, date: str) -> pd.Series:
        if EXCHANGE in ['CF', "CF5m"]:
            raise NotImplementedError("CF exchange not supported")
            # index = futures_metadata.index(date)
            # has_night_session = date in futures_metadata._no_night_session_dates()
            # if not has_night_session:
            #     return pd.Series(True, index=index)
            # day_index = index[(index.date == pd.to_datetime(date).date()) & (index.hour >= 9)]
            # mask = pd.Series(False, index=index)
            # mask.loc[day_index] = True  # day session
            # return mask
        else:
            return pd.Series(True, index=metadata.index(date))
    
    def _find_index(self, ts: float) -> int:
        i = np.searchsorted(self._full_index_ts, ts, side='right')
        assert isinstance(i, int) and i > 0
        # return i - 2
        return i - 1
    
    def _find_accounts(self) -> list[str]:
        return [p.parent.name for p in sdk.SHMEXE_ROOT.glob(f'*/{self._date}')]
    
    def _read_time_valid(self, i: int) -> pd.Series:
        assert EXCHANGE in ["okx5m", "binance5m"]
        return pd.Series(1, index=self._ctx.univ_symbols)

    def _is_trading_time(self, ts: float, index_ts: pd.Index) -> bool:
        i = np.searchsorted(index_ts, ts, side='left')
        assert isinstance(i, int)
        
        if i >= len(index_ts):
            return False  # after last index (including last index)
        if ts != index_ts[i]:
            return False  # not in the index
        return True

    def deploy_read_universe(self, universe_name: str) -> list[str]:
        if universe_name == 'all':
            return metadata.symbols(self._date)
        if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
            univ: pd.Series = dbtool.read_wide(self._date, f'source/meta/{universe_name}').squeeze()
            return univ.index[univ != 0].tolist()
        elif EXCHANGE in ["CF", "CF5m"]:
            raise NotImplementedError("CF exchange not supported")
            # univ = db.read(self._date, f'source/meta/{universe_name}').iloc[0]
            # return univ.index[univ != 0]
        else:
            raise ValueError("Unknown exchange: " + EXCHANGE)
            
    def deploy_read_last_holding(self, account) -> pd.Series:
        assert not self._net_mode
        ectx = self._ectxs[account]
        lp, sp = ectx.get_pos()
        assert isinstance(lp, dict) and isinstance(sp, dict)
        # filter universe
        univ_syms = self.deploy_read_universe(self._univ_name)
        lp = pd.Series({symbol: lp[symbol].qty if symbol in lp else 0.0 for symbol in univ_syms})
        sp = pd.Series({symbol: sp[symbol].qty if symbol in sp else 0.0 for symbol in univ_syms})
        net = lp - sp
        # calc notional
        i = self._find_index(time.time()) - self._full_index.shape[0] + self._index.shape[0]
        close = self._ctx.read_source(self._source_map['close_price'], i)
        notional = (net * close).fillna(0.0)
        return notional / self.deploy_read_nav(account)
    
    def read_valid(self, i: int) -> pd.Series:
        res = self._time_valid.loc[i]
        assert isinstance(res, pd.Series)
        return res
    
    def deploy_read_last_alpha(self, time_str, inst, alpha) -> pd.Series:
        last_alpha = self.last_alpha_cache.get((time_str, alpha), None)
        if last_alpha is not None and inst in last_alpha:
            return pd.Series(last_alpha[inst])
        ts = pd.Timestamp(time_str, tz='Asia/Shanghai').timestamp()
        i = self._find_index(ts) - self._full_index.shape[0] + self._index.shape[0]
        if alpha == 'valid':
            if EXCHANGE in ["okx", "okx10m", "binance5m", "okx5m"]:
                return self.deploy_read_last_alpha(time_str, inst, "ret1m").notna().astype(float)
            # return pd.Series(float(self.read_valid(i)[inst]))
            last_alpha = self.read_valid(i).astype(float)
        if inst.startswith('market_'):
            univ = inst[len('market_'):]
            # todo: 缓存加速
            return pd.Series(self._ctx.read_feature(f'market_{alpha}_{univ}', i)[0])
        if alpha in self._source_map:
            last_alpha = self._ctx.read_source(self._source_map[alpha], i)
        if last_alpha is None:
            last_alpha = self._ctx.read_feature(alpha, i)
        # return pd.Series(self._ctx.read_feature(alpha, i)[inst])
        self.last_alpha_cache[(time_str, alpha)] = last_alpha
        return pd.Series(last_alpha[inst])

    def deploy_read_nav(self, account: str) -> float:
        return self._ectxs[account].sh.read('nav').query('account == @account.encode()').nav.iloc[-1]

    def is_trading_time(self, time_str: str) -> bool:
        ts = pd.Timestamp(time_str, tz='Asia/Shanghai').timestamp()
        return self._is_trading_time(ts, self._full_index_ts)

    def deploy_read_fee_rates(self, account: str) -> pd.Series:
        ectx = self._ectxs[account]
        frs = pd.Series(ectx.get_fee_rates())
        return frs.loc[self.deploy_read_universe(self._univ_name)]

    def deploy_read_last_turnover(self, _, inst) -> pd.Series:
        i = self._find_index(time.time()) - self._full_index.shape[0] + self._index.shape[0]
        return pd.Series(self._ctx.read_source(self._source_map['turnover'], i)[inst])

    def deploy_write_holding(self, time_str, holding) -> None:
        t = pd.Timestamp(time_str, tz='Asia/Shanghai')
        index_ts = t.timestamp()
        holding_srs: pd.Series = holding['holding']
        for ectx in self._ectxs.values():
            for inst, w in holding_srs.items():
                if EXCHANGE in ["CF", "CF5m"]:
                    asset_type = sdk.AssetType.COMMODITY_FUTURES
                    # side = sdk.OrderSideType.BUY if w > 0 else sdk.OrderSideType.SELL
                else:
                    asset_type = sdk.AssetType.FUTURES

                side = sdk.Side.BUY if w > 0 else sdk.Side.SELL
                price = self.deploy_read_last_alpha(time_str, inst, 'close_price')
                predict_ts = index_ts
                if EXCHANGE in ["CF", "CF5m"] and t.second != 0:
                    # 对 20:59:30 和 08:59:30 这类提前了 index 的点，调整 fence_ts
                    predict_ts = index_ts + (60 - t.second)

                ectx.append_ep(inst, asset_type, side, abs(w), price, is_weight=True, index_ts=index_ts, predict_ts=predict_ts)

    def deploy_read_history_alpha(self, time_str, inst, alpha) -> pd.Series:
        ts = pd.Timestamp(time_str, tz='Asia/Shanghai').timestamp()
        end_i = self._find_index(ts) - self._full_index.shape[0] + self._index.shape[0] - 1
        start_i = end_i - (1024 - 1)
        if alpha == 'valid':
            if EXCHANGE in ["okx", "okx10m", "binance5m", "okx5m"]:
                return self.deploy_read_history_alpha(time_str, inst, "ret1m").notna().astype(float)
            price_valid = self.deploy_read_history_alpha(time_str, inst, "price_valid").astype(int)
            return pd.Series(self._time_valid.loc[start_i:end_i][inst].values & price_valid.values).astype(float)
        if EXCHANGE in ["CF", "CF5m"]:
            raise NotImplementedError("CF exchange not supported")
            # if inst.startswith('market_'):
            #     raise NotImplementedError
            # g = self._symbols_group[inst]
            # if alpha in self._source_map:
            #     select = -self._ctx.ctxs[g].daylen + self._ctx.day_index_map_ffill[g].loc[start_i:end_i].astype(int)
            #     return pd.Series(self._ctx.ctxs[g].source_dfs[f'source/{self._source_map[alpha]}'].iloc[select][inst].values)
            # nan_select = -self._ctx.ctxs[g].daylen + self._ctx.day_index_map[g].loc[start_i:end_i]
            # select = nan_select.fillna(0).astype(int)
            # return pd.Series(np.where(nan_select.isna(), 0, self._ctx.ctxs[g].feature_dfs[f'feature/{alpha}'].iloc[select][inst].values)).fillna(0)
        elif EXCHANGE in ["okx5m", "okx10m", "binance5m"]:
            if inst.startswith('market_'):
                univ = inst[len('market_'):]
                return pd.Series(self._ctx.feature_dfs[f'feature/market_{alpha}_{univ}'].iloc[-self._ctx.daylen + start_i:-self._ctx.daylen + end_i + 1][0].values)
            if alpha in self._source_map:
                return pd.Series(self._ctx.source_dfs[f'source/{self._source_map[alpha]}'].iloc[-self._ctx.daylen + start_i:-self._ctx.daylen + end_i + 1][inst].values)
            return pd.Series(self._ctx.feature_dfs[f'feature/{alpha}'].iloc[-self._ctx.daylen + start_i:-self._ctx.daylen + end_i + 1][inst].values)
        else:
            raise NotImplementedError


class Account:

    def __init__(self, cfg, parent_cfg_path, hist_row_csv_path, cli: SDKWrapper):
        self.account_name = cfg["name"]
        self.strategy_cfg = cfg["strategy"]
        self.stage = len(cfg["strategy"]["max_beta_exposure"])
        self.engles_cli = cli
        
        if hist_row_csv_path is None:
            self.hist_row = [pd.Series(np.nan) for _ in range(self.stage)]
        elif hist_row_csv_path == "deploy":
            self.hist_row = [self.engles_cli.deploy_read_last_holding(self.account_name) / self.stage for _ in range(self.stage)]

        elif hist_row_csv_path == "last":
            last_path_prefix = f"deploy/data/{self.account_name}"
            last_path_dir = sorted(os.listdir(last_path_prefix))[-1]
            last_filename = sorted([i for i in os.listdir(os.path.join(last_path_prefix, last_path_dir)) if i.endswith(".csv")])[-1]
            last_file_path = os.path.join(last_path_prefix, last_path_dir, last_filename)
            print(f"[INFO] read last holding from {last_file_path}")
            df = pd.read_csv(last_file_path, index_col=0)
            self.hist_row = [df.loc[f"holding_stage{i}(%)"] / 100 for i in range(self.stage)]
            assert f"holding_stage{self.stage}(%)" not in df.index
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

    def try_reset(self, task_date):
        curr_nav = self.engles_cli.deploy_read_nav(self.account_name)
        
        if EXCHANGE in ["CF", "CF5m"]:
            slip_time = "20:01"
        elif EXCHANGE in ["okx5m", "okx10m", "binance5m"]:
            slip_time = "08:01"
        else:
            raise ValueError("unknown exchange")
        if curr_nav != self.nav or self.curr_date != task_date:
            # if curr_nav != self.nav:
            #     self.reset_holding()
            self.curr_date = task_date
            self.nav = curr_nav
            self.run_name = f"{self.account_name}/{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
            os.makedirs(f"deploy/data/{self.run_name}", exist_ok=True)
            os.system(f"cp \"{self.parent_cfg_path}\" \"deploy/data/{self.run_name}/config.json5\"")
            os.system(f"echo \"{curr_nav}\" > \"deploy/data/{self.run_name}/nav.txt\"")
            print(f"run_name: {self.run_name}")


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
        new_ewm_var = ewm_alpha * ((curr_ret1m - new_ewm_mean)**2) + (1 - ewm_alpha) * ewm_var
        ewm_mean[valid] = new_ewm_mean[valid]
        ewm_var[valid] = new_ewm_var[valid]
        curr_vola = np.log1p(ewm_var**0.5)
    else:
        curr_vola = 1
    pred = label_std * pred * curr_vola
    pred = np.expm1(np.abs(pred)) * np.sign(pred)
    pred[~valid] = 0
    return pred, ewm_mean, ewm_var


def do_minute_infer(task_time_str, cfg, universe, model_name, model_id, engles_cli: SDKWrapper):
    try:
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
        
        if EXCHANGE in ["CF5m", "binance5m"]:
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
        elif EXCHANGE in ["okx5m", "okx10m"]:
            market_feature_name += [
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
        else:
            raise ValueError(f"Unknown exchange: {EXCHANGE}")

        # 读历史数据
        # history_inst_feature = [engles_cli.deploy_read_history_alpha(task_time_str, inst, inst_feature_name) for inst in universe]
        history_inst_self_feature_list = []
        for inst in universe:
            f = []
            for name in inst_feature_name:
                f.append(engles_cli.deploy_read_history_alpha(task_time_str, inst, name).rename(name))
            history_inst_self_feature_list.append(pd.concat(f, axis=1))
            
        if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
            history_market_feature = [engles_cli.deploy_read_history_alpha(task_time_str, market, "ret1m").rename(f"{market}_ret1m") for market in market_list]
            history_market_feature += [engles_cli.deploy_read_history_alpha(task_time_str, market_list[0], alpha).rename(f"{market_list[0]}_{alpha}") for alpha in market_feature_name]
            history_market_feature = pd.concat(history_market_feature, axis=1)
            history_index = history_market_feature.index

            history_inst_feature_list = []
            for inst_feature in history_inst_self_feature_list:
                history_inst_feature_list.append(pd.concat([inst_feature, history_market_feature], axis=1)[all_feature_name_list].values.T)

            history_inst_feature = np.array(history_inst_feature_list)  # B = 20, C = 18, T = 1439
            history_inst_ret1m = history_inst_feature[:, 0:1, :]  # 第一个 feature 必须是 ret1m！
            history_inst_ret1m = history_inst_ret1m[..., -(model.seq_len - 1):]

            norm_history_inst_feature = (history_inst_feature / normalizer["feature"].std[None, :, None]).clip(-normalizer["feature"].clip, normalizer["feature"].clip)
            norm_history_inst_feature = norm_history_inst_feature[..., -(model.seq_len - 1):]
            print(f"[{model_id}] to read history cost {(time.time() - st_time)*1e3:.2f}ms")

        if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
            valid = pd.concat([engles_cli.deploy_read_history_alpha(task_time_str, inst, "ret1m").notna().rename("valid") for inst in universe], axis=1).values
        elif EXCHANGE in ["CF5m"]:
            valid = pd.concat([engles_cli.deploy_read_history_alpha(task_time_str, inst, "valid").eq(1).rename("valid") for inst in universe], axis=1).values
        else:
            raise ValueError(f"Unknown exchange: {EXCHANGE}")
        
         # 设置状态
        for h in horizon:
            # 训练集 gen_data 里有 max(10, h)
            # 这里做检查
            # assert h >= 10
            if h < 10:
                print(f"======= [WARN] horizon < 10 =======")
        horizon = [max(10, h) for h in horizon]
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
                    encode_pred[:, c], history_inst_ret1m[..., i:i + 1], ewm_alpha[c], ewm_mean[c],
                    ewm_var[c], normalizer["label"].std[c:c + 1], valid[i], use_vola)
                preds[c].append(pred)
                prob = encode_prob[:, c] * 2 - 1
                probs[c].append(prob)
        # hist_df = pd.DataFrame(preds, index=history_index[-(model.seq_len - 1):], columns=universe)
        
        # 读最新数据
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "model read last alpha begin")
        read_last_st = time.time()
        last_inst_self_feature_list = []
        for inst in universe:
            f = []
            for name in inst_feature_name:
                f.append(engles_cli.deploy_read_last_alpha(task_time_str, inst, name).rename(name))
            last_inst_self_feature_list.append(pd.concat(f, axis=1))

        if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
            last_market_feature = [engles_cli.deploy_read_last_alpha(task_time_str, market, "ret1m").rename(f"{market}_ret1m") for market in market_list]
            last_market_feature += [engles_cli.deploy_read_last_alpha(task_time_str, market_list[0], alpha).rename(f"{market_list[0]}_{alpha}") for alpha in market_feature_name]
            last_market_feature = pd.concat(last_market_feature, axis=1)
            last_index = last_market_feature.index

        last_inst_feature = []
        for inst_feature in last_inst_self_feature_list:
            if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
                last_inst_feature.append(pd.concat([inst_feature, last_market_feature], axis=1)[all_feature_name_list].values.T)
            elif EXCHANGE in ["CF5m"]:
                last_inst_feature.append(pd.concat([inst_feature], axis=1)[all_feature_name_list].values.T)
            else:
                raise ValueError(f"Unknown exchange: {EXCHANGE}")
        last_inst_feature = np.array(last_inst_feature)  # B = 20, C = 18, T = 1
        last_inst_ret1m = last_inst_feature[:, 0:1, :]  # 第一个 feature 必须是 ret1m！
        norm_last_inst_feature = (last_inst_feature / normalizer["feature"].std[None, :, None]).clip(-normalizer["feature"].clip, normalizer["feature"].clip)
        if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
            last_valid = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "ret1m").notna().values[0] for inst in universe])
        elif EXCHANGE in ["CF5m"]:
            last_valid = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "valid").eq(1).values[0] for inst in universe])
        else:
            raise ValueError(f"Unknown exchange: {EXCHANGE}")

        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "model read last alpha begin")

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

        last_preds = [pd.Series(p, index=universe) for p in last_preds]
        last_probs = [pd.Series(p, index=universe) for p in last_probs]
        # print(f"[{model_id}] model cost {(time.time() - st_time)*1e3:.2f}ms")
        print(f"[{model_id}] read last cost {(time.time() - read_last_st)*1e3:.2f}ms")
        # st_time = time.time()
        # print(f"[{model_id}] opt cost {(time.time() - st_time)*1e3:.2f}ms")

    except Exception as e:
        dump_log(f"[{model_name} {model_id} ERROR], {e}", traceback.format_exc())
        return None
        last_preds = [pd.Series(0.0, index=universe) for _ in horizon]
        last_probs = [pd.Series(0.0, index=universe) for _ in horizon]

    return last_preds, last_probs


def do_minute_opt(
    cfg, last_row, signals, probs, last_price, last_std, last_turnover, last_bid1_price, last_ask1_price, last_bid1_volume,
    last_ask1_volume, last_valid, turnover_ma0, turnover_ma1,
    # turnover_ma2,
    corr1, corr2, corr3, oi, funding_fee, half_spread_mean, book1_value_sum0, book1_value_sum1, ret1m, ts, fee):

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
        dump_log(f"[OPT ERROR]", traceback.format_exc())
        return None


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json5.load(f)

    model_num = assert_mdl_conf(cfg)

    IS_TEST = args.is_test

    if IS_TEST:
        raise NotImplementedError("Test mode not implemented")
        # current_time = time.struct_time((2024, 2, 1, 9, 10, 45, 0, 0, 0))
        # while True:
        #     TMP_DATE = "2024-02-01"
        #     if current_time.tm_sec >= 40:
        #         current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        #         task_time = pd.Timestamp(current_time_str).floor("min") + pd.Timedelta(minutes=1)
        #         task_time_str = task_time.strftime("%Y-%m-%d %H:%M:%S")
        #         # do_minute_infer(task_time_str, args, hist_row_list[0], 0)
        #         with Pool(model_num) as p:
        #             hist_row_list = p.starmap(do_minute_infer, [(task_time_str, args, hist_row_list[model_id], model_id) for model_id in range(model_num)])
        #         deploy_write_holding(task_time_str, sum(hist_row_list) / model_num)
        #         current_time = time.localtime(time.mktime(current_time) + 60)
    else:

        while True:
            current_time = time.localtime()
            current_date = time.strftime("%Y-%m-%d", current_time)
            if current_time.tm_sec >= 5:

                check_print_file_handles()
                
                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
                task_time = pd.Timestamp(current_time_str).floor("30s") + pd.Timedelta(seconds=30)
                task_time_str = task_time.strftime("%Y-%m-%d %H:%M:%S")
                trade_date = get_trade_date(task_time_str)
                if trade_date is None:
                    print(f"[{task_time_str}] not trade date, skip, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                    time.sleep(30 - (time.localtime().tm_sec) % 30 + 5)
                    continue
                
                # SDK_WRAPPER = SDKWrapper(date=trade_date, accounts=[a['name'] for a in cfg['account']], net_mode=False, univ_name=cfg["universe"])
                # SDK_WRAPPER = SDKWrapper(date=trade_date, net_mode=False, univ_name=cfg["universe"])
                engles_cli = SDKWrapper(date=trade_date, accounts=[cfg['account'][0]['name']], net_mode=False, univ_name=cfg["universe"])
                account_list = [Account(c, args.config, args.prev_data_csv_path, engles_cli) for c in cfg["account"]]
                
                # print(f"task begin time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))}.{int((t - int(t)) * 1e3):03d}")
                try:
                    for account in account_list:
                        account.try_reset(trade_date)
                    if not engles_cli.is_trading_time(task_time_str):
                        print(f"[{task_time_str}] not trading time, skip, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                        time.sleep(30 - (time.localtime().tm_sec) % 30 + 5)
                        continue

                    # if cfg["universe"].endswith("_2d"):
                    #     raise ValueError("universe configured should not end with _2d")
                    universe = engles_cli.deploy_read_universe(cfg["universe"])  # T30R20
                    if EXCHANGE in ['CF', 'CF5m']:
                        universe_2d = engles_cli.deploy_read_universe('all')  # all
                    else:
                        universe_2d = engles_cli.deploy_read_universe(f'{cfg["universe"]}_2d')  # 2d
                    # engles_cli.deploy_write_holding(task_time_str, pd.Series(0.0, index=universe)) # 紧急平仓打开这个
                    # exit(0) # 紧急平仓打开这个

                    valid_insts = pd.Series(1, index=universe)

                    # coef_df = pd.read_csv("deploy/cost.csv", index_col=0).reindex(universe)
                    task_list = []
                    for model_id in range(model_num):
                        task_list.append((task_time_str, cfg, universe, 'model', model_id))
                    # do_minute_infer(task_time_str, cfg, universe, "model", 0)
                    # embed()
                    with Pool(model_num) as p:
                        ret = p.starmap(do_minute_infer, task_list)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "model done")
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
                    if EXCHANGE in ["okx", "okx10m", "binance5m", "okx5m"]:
                        fee = pd.Series(cfg["account"][0]["strategy"]["fee"], index=universe).values
                    elif EXCHANGE in ["CF5m"]:
                        fee = engles_cli.deploy_read_fee_rates(account_list[0].account_name).reindex(universe).fillna(0).values
                    else:
                        raise ValueError("fee not supported")

                    last_inst_close_price = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "close_price").values[0] for inst in universe])
                    # last_inst_bid1_price = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "bid1_price").values[0] for inst in universe])
                    # last_inst_ask1_price = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "ask1_price").values[0] for inst in universe])
                    last_std = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, cfg["std_name"]).values[0] for inst in universe])
                    last_valid = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "valid").eq(1).values[0] for inst in universe])
                    ret1m = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "ret1m").values[0] for inst in universe])
                    if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
                        turnover_ma0 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "turnover_r1").values[0] for inst in universe])
                        turnover_ma1 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "turnover_r200").values[0] for inst in universe])
                        # turnover_ma2 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "turnover_r200").values[0] for inst in universe])
                        corr1 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, f"corr_market_{cfg['universe']}_r10").values[0] for inst in universe])
                        corr2 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, f"corr_market_{cfg['universe']}_r40").values[0] for inst in universe])
                        corr3 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, f"corr_market_{cfg['universe']}_r160").values[0] for inst in universe])
                        oi = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "open_interest").values[0] for inst in universe])
                        funding_fee = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "funding_1hr_r24h").values[0] for inst in universe])
                    else:
                        turnover_ma0 = None
                        turnover_ma1 = None
                        # turnover_ma2 = None
                        corr1 = None
                        corr2 = None
                        corr3 = None
                        oi = None
                        funding_fee = None

                    # spread = (last_inst_ask1_price - last_inst_bid1_price)
                    # cost = spread * coef_coef / last_inst_close_price
                    # cost = coef_df["cost"] / 1e4
                    # MISSING_COST = cfg["cost_adjust"]["missing_cost"]
                    # MIN_COST = cfg["cost_adjust"]["min_cost"]
                    # MAX_COST = cfg["cost_adjust"]["max_cost"]
                    # # pass_cost = coef_df["cost"] / 1e4
                    # # assert cfg["cost_adjust"]["pct_shift"] <= 0.5 and cfg["cost_adjust"]["pct_shift"] >= 0
                    # # cost = cost.clip(pass_cost * (1 - cfg["cost_adjust"]["pct_shift"]), pas
                    # s_cost * (1 + cfg["cost_adjust"]["pct_shift"]))
                    # cost[~np.isfinite(cost)] = MISSING_COST
                    # cost = cost.clip(MIN_COST, MAX_COST)

                    assert len(account_list) == 1
                    for account in account_list:
                        if account.strategy_cfg["exec_info"]["exec_type"] in ["make", "make2"] and last_turnover is None:
                            last_turnover = np.array([engles_cli.deploy_read_last_turnover(task_time_str, inst).values[0] for inst in universe])
                            last_turnover[~np.isfinite(last_turnover)] = 0
                            half_spread_mean = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "half_spread_mean").values[0] for inst in universe])
                            book1_value_sum0 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "book1_value_sum_1s_r1").values[0] for inst in universe])
                            book1_value_sum1 = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "book1_value_sum_1s_r200").values[0] for inst in universe])
                        elif account.strategy_cfg["exec_info"]["exec_type"] == "take" and last_bid1_price is None and last_ask1_price is None and last_bid1_volume is None and last_ask1_volume is None:
                            last_bid1_price = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "bid1_price").values[0] for inst in universe])
                            last_ask1_price = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "ask1_price").values[0] for inst in universe])
                            last_bid1_volume = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "bid1_volume").values[0] for inst in universe])
                            last_ask1_volume = np.array([engles_cli.deploy_read_last_alpha(task_time_str, inst, "ask1_volume").values[0] for inst in universe])
                            last_bid1_price[~np.isfinite(last_bid1_price)] = 0
                            last_ask1_price[~np.isfinite(last_ask1_price)] = 0
                            last_bid1_volume[~np.isfinite(last_bid1_volume)] = 0
                            last_ask1_volume[~np.isfinite(last_ask1_volume)] = 0
                        else:
                            raise ValueError("unknown exec type")

                        account.deploy_last_row = engles_cli.deploy_read_last_holding(account.account_name).reindex(universe).fillna(0)
                        account.hist_row = [h.reindex(universe).fillna(0) for h in account.hist_row]
                        account.valid_insts = valid_insts.copy()

                        for l in account.hist_row:
                            account.valid_insts[l[l.abs() > 1e-6].index] = 1
                        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "opt begin")
                        account.fusion_row = do_minute_opt(
                            cfg=account.strategy_cfg, last_row=account.hist_row,
                            # signals=fusion_preds,
                            # probs=fusion_prob_preds,
                            signals=fusion_preds[:1], probs=fusion_prob_preds[:1],
                            last_price=last_inst_close_price, last_std=last_std,
                            last_turnover=last_turnover, last_bid1_price=last_bid1_price,
                            last_ask1_price=last_ask1_price, last_bid1_volume=last_bid1_volume,
                            last_ask1_volume=last_ask1_volume, last_valid=last_valid & account.valid_insts.eq(1).values,
                            turnover_ma0=turnover_ma0, turnover_ma1=turnover_ma1,
                            # turnover_ma2=turnover_ma2,
                            corr1=corr1, corr2=corr2, corr3=corr3, oi=oi, ret1m=ret1m, ts=task_time_str, fee=fee,
                            funding_fee=funding_fee, half_spread_mean=half_spread_mean, book1_value_sum0=book1_value_sum0, book1_value_sum1=book1_value_sum1,
                        )

                        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "opt done")
                        fusion_row_df = pd.DataFrame(
                            [sum(account.fusion_row).values, np.zeros_like(account.fusion_row[0]), last_inst_close_price],
                            index=["holding", "ty", "close_price"], columns=account.fusion_row[0].index).T
                        
                        engles_cli.deploy_write_holding(task_time_str, fusion_row_df)
                        
                        # write close holding
                        close_symbols = sorted(set(universe_2d) - set(universe))
                        if len(close_symbols) > 0:
                            close_holding = pd.Series(0.0, index=close_symbols).to_frame(name='holding')
                            engles_cli.deploy_write_holding(task_time_str, close_holding)
                        account.hist_row = account.fusion_row

                    t = time.time()
                    # holding 已传给执行，这里做一些信息记录和后处理
                    write_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
                    last_ret1ms = pd.Series(ret1m, index=universe).mul(1e4).rename("ret1m(bp)")
                    last_midret1ms = pd.Series([engles_cli.deploy_read_last_alpha(task_time_str, inst, "midret1m").values[0] for inst in universe], index=universe).mul(1e4).rename("midret1m(bp)")
                    if EXCHANGE in ["CF"]:
                        vwap_slippage = pd.Series([engles_cli.deploy_read_last_alpha(task_time_str, inst, "vwap_slippage").values[0] for inst in universe],
                                                  index=universe).mul(1e4).rename("vwap_slippage(bp)")
                    elif EXCHANGE in ["okx10m", "binance5m", "binance5m", "okx5m", "CF5m"]:
                        vwap_slippage = pd.Series([engles_cli.deploy_read_last_alpha(task_time_str, inst, "twap_slippage").values[0] for inst in universe],
                                                  index=universe).mul(1e4).rename("vwap_slippage(bp)")
                    else:
                        raise ValueError("unknown exchange")

                    # pred1 = fusion_pred1.mul(1e4).rename("pred1(bp)")
                    # pred2 = fusion_pred2.mul(1e4).rename("pred2(bp)")
                    preds = [p.mul(1e4).rename(f"pred{i+1}(bp)") for i, p in enumerate(fusion_preds)]
                    prob_preds = [p.rename(f"prob_pred{i+1}") for i, p in enumerate(fusion_prob_preds)]
                    for account in account_list:
                        info = pd.concat(
                            [
                                sum(account.fusion_row).rename("holding(%)").mul(100).to_frame(),
                                *[p.to_frame() for p in preds],
                                *[p.to_frame() for p in prob_preds],
                                last_ret1ms.to_frame(),
                                last_midret1ms.to_frame(),
                                vwap_slippage.to_frame(),
                                # vwap_slippage_mid.to_frame(),
                            ],
                            axis=1).T
                        info.loc["limit_make"] = False
                        info.loc["close_price"] = last_inst_close_price
                        info.loc["real_holding(%)"] = account.deploy_last_row.mul(100)
                        info.loc["valid"] = last_valid & account.valid_insts.eq(1)
                        if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
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
                        if account.strategy_cfg["exec_info"]["exec_type"] in ["make", "make2"]:
                            info.loc["last_turnover(1e6)"] = last_turnover / 1e6
                            info.loc["half_spread_mean/close(bp)"] = half_spread_mean / last_inst_close_price * 1e4
                        elif account.strategy_cfg["exec_info"]["exec_type"] == "take":
                            info.loc["last_bid1_price"] = last_bid1_price
                            info.loc["last_ask1_price"] = last_ask1_price
                            info.loc["last_bid1_volume"] = last_bid1_volume
                            info.loc["last_ask1_volume"] = last_ask1_volume
                            info.loc["last_bid1_turnover"] = last_bid1_price * last_bid1_volume
                            info.loc["last_ask1_turnover"] = last_ask1_price * last_ask1_volume
                            vwap_slippage_mid = pd.Series([engles_cli.deploy_read_last_alpha(task_time_str, inst, "vwap_slippage_mid").values[0] for inst in universe],
                                                          index=universe).mul(1e4).rename("vwap_slippage_mid(bp)")
                            info.loc["vwap_slippage_mid(bp)"] = vwap_slippage_mid
                        else:
                            raise ValueError("unknown exec type")

                        info.loc["last_var(1e-5)"] = last_std * last_std * 1e5
                        # exchange_api = create_exchange_api(account.account_name)
                        # real_qty_holding = pd.Series(exchange_api.query_position())
                        # info.loc["real_qty_holding"] = real_qty_holding.reindex(info.columns).fillna(0)
                        # info.loc["real_equity"] = exchange_api.query_account_information().margin_balance
                        info.loc["real_equity"] = deploy_read_equity(engles_cli, account.account_name)
                        for i, h in enumerate(account.hist_row):
                            info.loc[f"holding_stage{i}(%)"] = h.mul(100)
                        info = info.round(3)
                        tmp = info.copy()
                        if EXCHANGE in ["binance5m", "okx5m", "okx10m"]:
                            tmp.columns = [i.split("-")[1] for i in tmp.columns]
                        elif EXCHANGE in ["CF", "CF5m"]:
                            pass
                        else:
                            raise ValueError("unknown exchange")
                        print(tmp.round(3))
                        print(f"holding sum: {info.loc['real_holding(%)'].sum():.3f}%")
                        print(f"holding abs sum: {info.loc['real_holding(%)'].abs().sum():.3f}%")
                        info.to_csv(f"deploy/data/{account.run_name}/{task_time_str}.csv")

                    # for account in account_list:
                    #     os.system(f"python deploy/show_pnl.py '{account.run_name}'")
                    print(task_time_str, "done")
                    t = time.time()
                    task_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
                    print(f"write time: {write_time_str}, end time: {task_end_time_str}")

                except Exception as e:
                    # engles_cli.deploy_write_holding(task_time_str, sum(hist_row_list) / model_num)
                    dump_log("[MAIN ERROR]", time.strftime("%Y-%m-%d %H:%M:%S", current_time), traceback.format_exc())
                    # for account in account_list:
                    #     account.reset_holding()
                finally:
                    engles_cli.close()
