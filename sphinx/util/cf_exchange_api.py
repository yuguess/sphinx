from multiprocessing.sharedctypes import Value
from tensorboard.compat.tensorflow_stub.errors import UnimplementedError
import pandas as pd

from mona.common import IS_CS_STORAGE, db, metadata_cf
from mona.common import DATE_PATTERN
from mona.common.util import dbtool
from sphinx.base_adt import DateS, DateS_L, SymS_L, UnivS, AlphaS, PanelDF, SymS
from .runtime_config import get_env_freq, get_env_exchange


CF_DEFAULT_UNIVERSE = "universe_t55r1_oi_CF"


def get_index(date: DateS) -> pd.DatetimeIndex:
    return metadata_cf.FuturesMetadata.index(date)


def get_dates(start_date: DateS, end_date: DateS) -> DateS_L:
    if not DATE_PATTERN.match(start_date) or not DATE_PATTERN.match(end_date):
        raise ValueError(f"invalid date: {start_date} or {end_date}")
    return metadata_cf.dates(start_date, end_date)
    # """返回闭区间内的交易日字符串。"""
    # dates = db.dates(f"source/meta/{CF_DEFAULT_UNIVERSE}")
    # return [date for date in dates if start_date <= date <= end_date]


def today_all_inst(date: DateS) -> SymS_L:
    return metadata_cf.symbols(date)


def read_universe(date: DateS, univ_name: UnivS) -> pd.Series:
    univ: pd.Series = dbtool.read_wide(date, f'source/meta/{univ_name}').squeeze()
    return pd.Series(0.0, index=univ.index[univ != 0])


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


def read_valid_frame(date: str, insts) -> pd.DataFrame:
    return read_feature_frame(date, insts, "valid").eq(1)


def read_feature_frame(date: DateS, syms: pd.Index, ftr_nm: AlphaS) -> PanelDF:
    univ = pd.Index(syms)
    freq = get_env_freq()
    key = f"{freq}/feature/{ftr_nm}"
    return db.read(date, key).reindex(columns=univ)


def _read_field(date: str, group: str, field: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    # value = _wide(date, )
    freq = get_env_freq()
    key = f"{freq}/source/{group}/{field}"
    value = db.read(date, key)
    return value.reindex(index)


def read_basedata(date: str, inst: str) -> pd.DataFrame:
    index = get_index(date)
    fields = {field: _read_field(date, "kline", field, index)[inst] for field in ["close", "volume", "turnover"]}
    return pd.DataFrame(fields, index=index).fillna(0)


def sample_per_date() -> int:
    exchange = get_env_exchange()
    if exchange == "CF":
        return 559
    elif exchange == "CF5m":
        return 115
    else:
        raise ValueError(f"unsupported sample_per_date for EXCHANGE={exchange}")
    
    
def read_orderbook(date: DateS, inst: SymS) -> pd.DataFrame:
    raise ValueError("implement later")
