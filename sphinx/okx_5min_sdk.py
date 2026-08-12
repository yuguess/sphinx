import time
import math
import pandas as pd
import numpy as np

from mona import sdk
from mona.common.util import dbtool


def is_trading_time(dt_s, time_str) -> bool:
    _full_index = sdk.metadata.index_with_history(dt_s, sdk.DEPLOY_HISTORY_LENGTH)
    _full_index_ts = _full_index.astype(int) / 1e9
    ts = pd.Timestamp(time_str, tz='Asia/Shanghai').timestamp()
    i = np.searchsorted(_full_index_ts, ts, side='left')
    if i >= len(_full_index_ts):
        return False  # after last index (including last index)
    if ts != _full_index_ts[i]:
        return False  # not in the index
    return True


class SDKWrapper:

    def __init__(self, date: str, net_mode=False, univ_name=None, accounts: list[str] = None) -> None:
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

        # days_num = math.ceil(sdk.DEPLOY_HISTORY_LENGTH / self._index.shape[0])
        # dates = metadata.dates(metadata.calc_date(self._date, -days_num), self._date)

        self._index_groups = None
        self._group_index_ts = None
        self._symbols_group = None

        relative_index = (self._index.shape[0] - np.arange(self._full_index.shape[0]) - 1)[::-1]

        self._time_valid = pd.concat([self._read_time_valid(i) for i in relative_index], axis=1).T
        self._time_valid.index = relative_index
        # apply night session mask if CF
        self._time_valid_normal = self._time_valid.copy()
        self.last_alpha_cache = {}

    def _read_time_valid(self, i: int) -> pd.Series:
        return pd.Series(1, index=self._ctx.univ_symbols)

    def _find_accounts(self) -> list[str]:
        return [p.parent.name for p in sdk.SHMEXE_ROOT.glob(f'*/{self._date}')]

    def _is_trading_time(self, ts: float, index_ts: pd.Index) -> bool:
        i = np.searchsorted(index_ts, ts, side='left')
        if i >= len(index_ts):
            return False  # after last index (including last index)
        if ts != index_ts[i]:
            return False  # not in the index
        return True

    def _find_index(self, ts: float) -> int:
        i = np.searchsorted(self._full_index_ts, ts, side='right')
        assert i > 0
        # return i - 2
        return i - 1

    def deploy_read_nav(self, account: str) -> float:
        return self._ectxs[account].sh.read('nav').query('account == @account.encode()').nav.iloc[-1]

    def is_trading_time(self, time_str: str) -> bool:
        ts = pd.Timestamp(time_str, tz='Asia/Shanghai').timestamp()
        return self._is_trading_time(ts, self._full_index_ts)

    def deploy_read_universe(self, universe_name: str) -> list[str]:
        univ: pd.Series = dbtool.read_wide(self._date, f'source/meta/{universe_name}').squeeze()
        return univ.index[univ != 0].tolist()

    def deploy_read_history_alpha(self, time_str, inst, alpha) -> pd.Series:
        ts = pd.Timestamp(time_str, tz='Asia/Shanghai').timestamp()
        end_i = self._find_index(ts) - self._full_index.shape[0] + self._index.shape[0] - 1
        start_i = end_i - (1024 - 1)
        if alpha == 'valid':
            return self.deploy_read_history_alpha(time_str, inst, "ret1m").notna().astype(float)

        if inst.startswith('market_'):
            univ = inst[len('market_'):]
            return pd.Series(self._ctx.feature_dfs[f'feature/market_{alpha}_{univ}'].iloc[-self._ctx.daylen + start_i:-self._ctx.daylen + end_i + 1][0].values)
        if alpha in self._source_map:
            return pd.Series(self._ctx.source_dfs[f'source/{self._source_map[alpha]}'].iloc[-self._ctx.daylen + start_i:-self._ctx.daylen + end_i + 1][inst].values)
        return pd.Series(self._ctx.feature_dfs[f'feature/{alpha}'].iloc[-self._ctx.daylen + start_i:-self._ctx.daylen + end_i + 1][inst].values)

    def deploy_read_last_alpha(self, time_str, inst, alpha) -> pd.Series:
        last_alpha = self.last_alpha_cache.get((time_str, alpha), None)
        if last_alpha is not None and inst in last_alpha:
            return pd.Series(last_alpha[inst])
        ts = pd.Timestamp(time_str, tz='Asia/Shanghai').timestamp()
        i = self._find_index(ts) - self._full_index.shape[0] + self._index.shape[0]

        if alpha == 'valid':
            return self.deploy_read_last_alpha(time_str, inst, "ret1m").notna().astype(float)

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

    def deploy_read_last_holding(self, account) -> pd.Series:
        assert not self._net_mode
        ectx = self._ectxs[account]
        lp, sp = ectx.get_pos()
        # filter universe
        lp = pd.Series({symbol: lp[symbol].qty if symbol in lp else 0.0 for symbol in self.deploy_read_universe(self._univ_name)})
        sp = pd.Series({symbol: sp[symbol].qty if symbol in sp else 0.0 for symbol in self.deploy_read_universe(self._univ_name)})
        net = lp - sp
        # calc notional
        i = self._find_index(time.time()) - self._full_index.shape[0] + self._index.shape[0]
        close = self._ctx.read_source(self._source_map['close_price'], i)
        notional = (net * close).fillna(0.0)
        return notional / self.deploy_read_nav(account)

    def deploy_read_last_turnover(self, _, inst) -> pd.Series:
        i = self._find_index(time.time()) - self._full_index.shape[0] + self._index.shape[0]
        return pd.Series(self._ctx.read_source(self._source_map['turnover'], i)[inst])

    def deploy_read_equity(self, account: str) -> float:
        fund = self._ectxs[account].get_fund()
        return fund.eq

    def close(self):
        for ectx in self._ectxs.values():
            ectx.close()
        self._ctx.close()


def create_infra_sdk(trade_dt_s, cfg) -> SDKWrapper:
    sdk_wrapper = SDKWrapper(date=trade_dt_s, accounts=[cfg['account'][0]['name']], net_mode=False, univ_name=cfg["universe"])
    return sdk_wrapper
