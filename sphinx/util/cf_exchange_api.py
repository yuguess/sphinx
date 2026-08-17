import pandas as pd

from mona.common import IS_CS_STORAGE, db, metadata_cf
from mona.common.util import dbtool
from sphinx.base_adt import DateS, DateS_L, SymS_L, UnivS
from .runtime_config import get_env_freq

CF_DEFAULT_UNIVERSE = "universe_t55r1_oi_CF"


def get_index(date: DateS) -> pd.DatetimeIndex:
    return metadata_cf.FuturesMetadata.index(date)


def get_dates(start_date: DateS, end_date: DateS) -> DateS_L:
    """返回闭区间内的交易日字符串。"""
    dates = db.dates(f"source/meta/{CF_DEFAULT_UNIVERSE}")
    return [date for date in dates if start_date <= date <= end_date]


def today_all_inst(date: DateS) -> SymS_L:
    return metadata_cf.symbols(date)


def read_universe(date: DateS, univ_name: UnivS):
    univ: pd.Series = dbtool.read_wide(date, f'source/meta/{univ_name}').squeeze()
    return univ.index[univ != 0].tolist()


def next_date(date: DateS) -> DateS:
    return metadata_cf.next_date(date)


def prev_date(date: DateS) -> DateS:
    return metadata_cf.prev_date(date)


def read_alpha(date: str, inst: str, alpha_name: str) -> pd.Series:
    freq = get_env_freq()
    if inst.startswith('market_'):
        univ = inst[len('market_'):]
        key = f'market_{alpha_name}_{univ}'
        return db.read(date, f'{freq}/feature/{key}').squeeze()
    elif IS_CS_STORAGE:
        return db.read(date, f'{freq}/feature/{alpha_name}')[inst]
    else:
        return db.read(date, f'{freq}/feature/{alpha_name}/{inst}').squeeze()
    

def _read_field(date: str, group: str, field: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    # value = _wide(date, )
    key = f"1min/source/{group}/{field}"
    value = db.read(date, key)
    return value.reindex(index)

def read_basedata(date: str, inst: str) -> pd.DataFrame:
    index = get_index(date)
    fields = {field: _read_field(date, "kline", field, index)[inst] for field in ["close", "volume", "turnover"]}
    return pd.DataFrame(fields, index=index).fillna(0)