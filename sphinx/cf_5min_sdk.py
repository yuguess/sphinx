import math
import pandas as pd
import numpy as np
import time

# from mona import sdk
from mona import cf_legacy_sdk as sdk
from mona.common import db


class SDKWrapper:

    def __init__(self, date: str, net_mode: bool, univ_name: str, accounts: list[str]) -> None:
        self._date = date
        self._net_mode = net_mode
        self._ctx = sdk.SHMContext(date)
        self._univ_name = univ_name
        if accounts is None:
            accounts = self._find_accounts()
        if diff := set(accounts) - set(self._find_accounts()):
            raise ValueError(f"accounts {diff} not found")
        self._ectxs = {a: sdk.ExeSHMContext(date, a, net_mode=net_mode, write_safe=False) for a in accounts}
        self._date = date
        self._index = sdk.metadata.index(date)
        self._full_index = sdk.metadata.index_with_history(date, sdk.DEPLOY_HISTORY_LENGTH)
        self._full_index_ts = self._full_index.astype(int) / 1e9
        self._source_map = {
            "close_price": "kline/close",
            "turnover": "kline/turnover",
            "bid1_price": "quotes/bid1_price",
            "ask1_price": "quotes/ask1_price",
            "bid1_volume": "quotes/bid1_volume",
            "ask1_volume": "quotes/ask1_volume",
        }
        
        days_num = math.ceil(sdk.DEPLOY_HISTORY_LENGTH / self._index.shape[0])
        dates = metadata.dates(metadata.calc_date(self._date, -days_num), self._date)
        self._index_groups = futures_metadata.index_groups(self._date)
        self._group_index_ts = {
            g: pd.DatetimeIndex([i for date in dates for i in futures_metadata.index(date, security_group=g)]).astype(int) / 1e9
            for g in self._index_groups}
        self._symbols_group = futures_metadata.symbols_group(self._date)

        relative_index = (self._index.shape[0] - np.arange(self._full_index.shape[0]) - 1)[::-1]
        self._time_valid = pd.concat([self._read_time_valid(i) for i in relative_index], axis=1).T
        self._time_valid.index = relative_index

        # apply night session mask if CF
        self._time_valid_normal = self._time_valid.copy()

        full_night_session_mask = pd.concat([self._night_session_mask(date) for date in dates], axis=0).reindex(self._full_index)
        self._time_valid.loc[~full_night_session_mask.values] = False

        self.last_alpha_cache = {}
        
    def _night_session_mask(self, date: str) -> pd.Series:
        index = futures_metadata.index(date)
        has_night_session = date in futures_metadata._no_night_session_dates()
        if not has_night_session:
            return pd.Series(True, index=index)
        day_index = index[(index.date == pd.to_datetime(date).date()) & (index.hour >= 9)]
        mask = pd.Series(False, index=index)
        mask.loc[day_index] = True  # day session
        return mask

    def _find_accounts(self) -> list[str]:
        return [p.parent.name for p in sdk.SHMEXE_ROOT.glob(f"*/{self._date}")]
    
    def _find_index(self, ts: float) -> int:
        i = np.searchsorted(self._full_index_ts, ts, side="right")
        assert i > 0
        # return i - 2
        return i - 1
    
    def _read_time_valid(self, i: int) -> pd.Series:
        ts = self._full_index_ts[-self._index.shape[0] + i]
        
        group_time_valid = pd.Series(
            [self._is_trading_time(ts + 60, self._group_index_ts[g]) for g in self._index_groups],
            index=self._index_groups)
        time_valid = group_time_valid[self._symbols_group]
        time_valid.index = self._symbols_group.index
        return time_valid
    
    def _is_trading_time(self, ts: float, index_ts: pd.Index) -> bool:
        i = np.searchsorted(index_ts, ts, side="left")
        if i >= len(index_ts):
            return False  # after last index (including last index)
        if ts != index_ts[i]:
            return False  # not in the index
        return True
    
    def read_valid(self, i: int) -> pd.Series:
        return self._time_valid.loc[i]
    
    def deploy_read_universe(self, universe_name: str) -> list[str]:
        univ = db.read(self._date, f"source/meta/{universe_name}").iloc[0]
        return univ.index[univ != 0]
    
    def deploy_read_history_alpha(self, time_str, inst, alpha) -> pd.Series:
        ts = pd.Timestamp(time_str, tz="Asia/Shanghai").timestamp()
        end_i = (self._find_index(ts) - self._full_index.shape[0] + self._index.shape[0] - 1)
        start_i = end_i - (1024 - 1)
        if alpha == "valid":
            price_valid = self.deploy_read_history_alpha(time_str, inst, "price_valid").astype(int)
            return pd.Series(self._time_valid.loc[start_i:end_i][inst].values & price_valid.values).astype(float)

        if inst.startswith("market_"):
            raise NotImplementedError
        
        if inst.startswith('market_'):
            raise NotImplementedError
        g = self._symbols_group[inst]
        if alpha in self._source_map:
            select = -self._ctx.ctxs[g].daylen + self._ctx.day_index_map_ffill[g].loc[start_i:end_i].astype(int)
            return pd.Series(self._ctx.ctxs[g].source_dfs[f'source/{self._source_map[alpha]}'].iloc[select][inst].values)
        nan_select = -self._ctx.ctxs[g].daylen + self._ctx.day_index_map[g].loc[start_i:end_i]
        select = nan_select.fillna(0).astype(int)
        return pd.Series(np.where(nan_select.isna(), 0, self._ctx.ctxs[g].feature_dfs[f'feature/{alpha}'].iloc[select][inst].values)).fillna(0)
    
    def deploy_read_last_alpha(self, time_str, inst, alpha) -> pd.Series:
        last_alpha = self.last_alpha_cache.get((time_str, alpha), None)
        if last_alpha is not None and inst in last_alpha:
            return pd.Series(last_alpha[inst])
        ts = pd.Timestamp(time_str, tz="Asia/Shanghai").timestamp()
        i = self._find_index(ts) - self._full_index.shape[0] + self._index.shape[0]
        if alpha == "valid":
            # return pd.Series(float(self.read_valid(i)[inst]))
            last_alpha = self.read_valid(i).astype(float)
        if inst.startswith("market_"):
            univ = inst[len("market_") :]
            # todo: 缓存加速
            return pd.Series(self._ctx.read_feature(f"market_{alpha}_{univ}", i)[0])
        if alpha in self._source_map:
            last_alpha = self._ctx.read_source(self._source_map[alpha], i)
        if last_alpha is None:
            last_alpha = self._ctx.read_feature(alpha, i)
        # return pd.Series(self._ctx.read_feature(alpha, i)[inst])
        self.last_alpha_cache[(time_str, alpha)] = last_alpha
        return pd.Series(last_alpha[inst])

    def deploy_write_holding(self, time_str, holding) -> None:
        predict_ts = pd.Timestamp(time_str, tz="Asia/Shanghai").timestamp()
        holding: pd.Series = holding["holding"]
        for ectx in self._ectxs.values():
            for inst, w in holding.items():
                asset_type = sdk.AssetType.COMMODITY_FUTURES
                side = sdk.OrderSideType.BUY if w > 0 else sdk.OrderSideType.SELL
                price = self.deploy_read_last_alpha(time_str, inst, "close_price")
                ectx.append_ep(inst, asset_type, side, abs(w), is_weight=True, predict_ts=predict_ts)

    def deploy_read_last_holding(self, account) -> pd.Series:
        assert not self._net_mode
        ectx = self._ectxs[account]
        lp, sp = ectx.get_pos()
        # filter universe
        lp = pd.Series({symbol: lp[symbol].qty if symbol in lp else 0.0 for symbol in self.deploy_read_universe(self._univ_name)})
        sp = pd.Series({symbol: sp[symbol].qty if symbol in sp else 0.0 for symbol in self.deploy_read_universe(self._univ_name)})
        net = lp - sp
        # calc notional
        i = (self._find_index(time.time()) - self._full_index.shape[0] + self._index.shape[0])
        close = self._ctx.read_source(self._source_map["close_price"], i)
        notional = (net * close).fillna(0.0)
        return notional / self.deploy_read_nav(account)

    def deploy_read_nav(self, account: str) -> float:
        if account == "fu_prod_001":
            return 6000e4
        elif account == "okx_prod_test":
            return 5e4
        else:
            raise ValueError("Unknown account: " + account)

    def deploy_read_equity(self, account: str) -> float:
        fund = self._ectxs[account].get_adjusted_fund()
        return fund.eq

    def deploy_read_last_turnover(self, _, inst) -> pd.Series:
        i = (self._find_index(time.time()) - self._full_index.shape[0] + self._index.shape[0])
        return pd.Series(self._ctx.read_source(self._source_map["turnover"], i)[inst])

    def deploy_read_fee_rates(self, account: str) -> pd.Series:
        ectx = self._ectxs[account]
        frs = pd.Series(ectx.get_fee_rates())
        return frs.loc[self.deploy_read_universe(self._univ_name)]

    def close(self):
        for ectx in self._ectxs.values():
            ectx.close()
        self._ctx.close()


def create_infra_sdk(trade_dt_s, cfg):
    sdk_wrapper = SDKWrapper(date=trade_dt_s, accounts=[cfg['account'][0]['name']], net_mode=False, univ_name=cfg["universe"])
    return sdk_wrapper
