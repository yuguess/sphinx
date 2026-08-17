import getpass
import os
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
from typing import Any

from mona.common.util import dbtool
from mona.common import IS_CS_STORAGE, db, metadata
from .runtime_config import CF_EXCHANGE, FREQ_1H, FREQ_1S, FREQ_5MIN, OKX_EXCHANGE, get_env_exchange, get_env_freq, BINANCE5M
from .runtime_config import require_runtime_config


CF_DEFAULT_UNIVERSE = "universe_t55r1_oi_CF"
OKX_1S_INDEX_INST = "OKP-BTC-USDT"
OKX_1H_INDEX_INST = "OKP-BTC-USDT"


def assert_supported_runtime() -> None:
    exchange = get_env_exchange()
    freq = get_env_freq()
    market = os.environ.get("MARKET", "")
    if exchange == CF_EXCHANGE and freq != FREQ_5MIN:
        raise ValueError(f"only CF {FREQ_5MIN} is supported, got EXCHANGE={exchange!r} FREQ={freq!r}")
    require_runtime_config({"EXCHANGE": exchange, "FREQ": freq, "MARKET": market}, error_type=ValueError)


def get_dates(start_date: str, end_date: str) -> list[str]:
    """返回闭区间内的交易日字符串。"""
    assert_supported_runtime()
    exchange = get_env_exchange()
    if exchange == OKX_EXCHANGE or exchange == BINANCE5M:
        return metadata.dates(start_date, end_date)
    else:
        raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")


def get_index(date: str) -> pd.Index:
    """读取当前交易日当前频率的 bar index。"""
    exchange = get_env_exchange()
    freq = get_env_freq()
    if exchange == CF_EXCHANGE:
        return db.read(date, f"{get_env_freq()}/source/kline/close").index
    elif exchange == OKX_EXCHANGE or exchange == BINANCE5M:
        if freq == FREQ_5MIN:
            return metadata.index(date)
        elif freq == FREQ_1S:
            return db.read(date, f"{freq}/source/quotes/{OKX_1S_INDEX_INST}").index
        elif freq == FREQ_1H:
            return db.read(date, f"{freq}/source/kline/{OKX_1H_INDEX_INST}").index
        else:
            raise ValueError(f"unsupported OKX FREQ={freq!r}")
    else:
        raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")


def sample_per_date(date: str | None = None) -> int:
    """返回当前频率下单日 bar 数。"""
    exchange = get_env_exchange()
    freq = get_env_freq()

    if date is not None:
        return len(get_index(date))

    if exchange == OKX_EXCHANGE and freq == FREQ_5MIN:
        return 288
    elif exchange == BINANCE5M:
        return 288
    elif exchange == OKX_EXCHANGE and freq == FREQ_1S:
        return 86400
    elif exchange == OKX_EXCHANGE and freq == FREQ_1H:
        return 24
    elif exchange == CF_EXCHANGE and freq == FREQ_5MIN:
        return 115
    elif exchange == CF_EXCHANGE and freq != FREQ_5MIN:
        raise ValueError(f"only CF {FREQ_5MIN} is supported, got EXCHANGE={exchange!r} FREQ={freq!r}")
    else:
        raise ValueError(f"unsupported runtime EXCHANGE={exchange!r} FREQ={freq!r}")


def today_all_inst(date: str) -> list[str]:
    """返回当日 metadata 可交易合约列表。"""
    return metadata.symbols(date)


def _read_meta_wide(date: str, name: str) -> pd.Series | pd.DataFrame:
    return dbtool.read_wide(date, f"source/meta/{name}")


def read_universe(date: str, univ_name: str) -> pd.Series:
    """读取指定交易日的 universe。"""
    exchange = get_env_exchange()
    if exchange == OKX_EXCHANGE:
        univ = _read_meta_wide(date, univ_name).squeeze()
    elif exchange == BINANCE5M:
        univ = _read_meta_wide(date, univ_name).squeeze()
    elif exchange == CF_EXCHANGE:
        univ = db.read(date, f"source/meta/{univ_name}").squeeze()
    else:
        raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")
    return pd.Series(0.0, index=univ.index[univ != 0])


def _read_source_table(date: str, inst: str, source_name: str) -> pd.DataFrame:
    return db.read(date, f"{get_env_freq()}/source/{source_name}/{inst}")


def _read_feature(date: str, inst: str, feature_name: str) -> Any:
    if inst.startswith("market_"):
        univ_name = inst.removeprefix("market_")
        return db.read(date, f"{get_env_freq()}/feature/market_{feature_name}_{univ_name}").squeeze()
    elif IS_CS_STORAGE is True:
        return db.read(date, f"{get_env_freq()}/feature/{feature_name}")[inst]
    elif IS_CS_STORAGE is False:
        return db.read(date, f"{get_env_freq()}/feature/{feature_name}/{inst}").squeeze()
    else:
        raise ValueError(f"unexpected IS_CS_STORAGE={IS_CS_STORAGE!r}")


def read_market_data(date: str, inst: str, data_name: str) -> pd.Series | pd.DataFrame:
    """按底层数据名读取单合约行情或特征。
    支持的 data_name 示例：
    - "kline/close"、"kline/turnover"
    - "feature/twap_slippage"
    - "quotes/bid1_price"、"bps/ask_vol_1bp"
    """

    if data_name in {"orderbook", "basedata"} or data_name.startswith(("orderbook/", "basedata/")):
        raise ValueError(
            "read_market_data 不再支持 orderbook/basedata 逻辑名称；"
            "请直接使用底层 source，例如 kline/close、quotes/bid1_price 或 feature/twap_slippage"
        )

    if data_name.startswith("feature/"):
        feature_name = data_name.removeprefix("feature/")
        return _read_feature(date, inst, feature_name)

    exchange = get_env_exchange()
    freq = get_env_freq()
    if exchange == CF_EXCHANGE and data_name in {"kline/close", "kline/turnover", "kline/volume"}:
        column = data_name.rsplit("/", 1)[1]
        return db.read(date, f"{freq}/source/kline/{column}")[inst]
    elif exchange == OKX_EXCHANGE and data_name.startswith("kline/"):
        column = data_name.rsplit("/", 1)[1]
        return db.read(date, f"{freq}/source/kline/{inst}")[column]

    if exchange == OKX_EXCHANGE:
        if "/" in data_name:
            source_name, column = data_name.split("/", 1)
            source = _read_source_table(date, inst, source_name)
            return source[column]
        return _read_source_table(date, inst, data_name)
    elif exchange == CF_EXCHANGE:
        if "/" in data_name:
            source_name, column = data_name.split("/", 1)
            source = _read_source_table(date, inst, source_name)
            return source[column]
        return _read_source_table(date, inst, data_name)
    else:
        raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")


def read_alpha(date: str, inst: str, alpha_name: str) -> Any:
    """读取单合约 feature。"""
    return read_market_data(date, inst, f"feature/{alpha_name}")


def next_date(date: str) -> str:
    next_date_value = metadata.next_date(date)
    if next_date_value is None:
        raise ValueError(f"next_date not found for {date}")
    return next_date_value


def prev_date(date: str) -> str:
    prev_date_value = metadata.prev_date(date)
    if prev_date_value is None:
        raise ValueError(f"prev_date not found for {date}")
    return prev_date_value


def read_signal(date: str, univ_name: str, signal_name: str) -> pd.DataFrame:
    freq = get_env_freq()
    key = f"{freq}/signal/{univ_name}/{signal_name}"
    return db.read(date, key)


def read_holding(date: str, univ_name: str, holding_key: str) -> pd.DataFrame:
    # return db.read(date, f"user/{getpass.getuser()}/{get_env_freq()}/holding/{univ_name}/{holding_key}")
    return db.read(date, f"{get_env_freq()}/holding/{univ_name}/{holding_key}")


def write_holding(date: str, univ_name: str, holding_key: str, holding: pd.DataFrame) -> None:
    db.write(date, f"{get_env_freq()}/holding/{univ_name}/{holding_key}", holding, is_cross_section=True)


# def write_signal(date: str, univ_name: str, signal_key: str, signal: pd.DataFrame) -> None:
#     """写入 DB signal 宽表。"""
#     freq = get_env_freq()
#     db.write(date, f"{freq}/signal/{univ_name}/{signal_key}", signal, is_cross_section=True)


def write_signal(date: str, univ_name: str, signal_key: str, signal: pd.DataFrame) -> None:
    db.write(date, f"{get_env_freq()}/signal/{univ_name}/{signal_key}", signal, is_cross_section=True)


def del_holding(univ_name: str, holding_key: str) -> None:
    db.delete(f"user/{getpass.getuser()}/{get_env_freq()}/holding/{univ_name}/{holding_key}")


def fee_cache_ok_mask(value: pd.Series) -> pd.Series:
    """解析 fee cache 里的可选 ok 列。"""

    if value.dtype == bool:
        return value
    if pd.api.types.is_numeric_dtype(value):
        return value != 0
    text = value.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


@lru_cache(maxsize=32)
def load_fee_cache(path: Path) -> pd.Series:
    """加载并校验本地 fee cache。"""

    if not path.exists():
        raise ValueError(f"fee_cache not found: {path}")
    cache = pd.read_csv(path, dtype={"date": str, "inst": str})
    required = {"date", "inst", "fee_rate"}
    missing = required.difference(cache.columns)
    if missing:
        raise ValueError(f"fee_cache missing columns: {sorted(missing)}")
    if "ok" in cache.columns:
        cache = cache[fee_cache_ok_mask(cache["ok"])]
    if cache[["date", "inst"]].duplicated().any():
        duplicate = cache[cache[["date", "inst"]].duplicated()][["date", "inst"]].iloc[0]
        raise ValueError(f"fee_cache duplicate row: {duplicate['date']} {duplicate['inst']}")
    fee = cache.set_index(["date", "inst"])["fee_rate"].astype(float)
    if not np.isfinite(fee.to_numpy()).all() or (fee < 0.0).any():
        raise ValueError("fee_cache.fee_rate must be finite and non-negative")
    return fee


def _require_finite_nonnegative_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def read_fee_rates_from_cache(path: Path, date: str, columns: pd.Index) -> pd.Series:
    """从本地 CSV 缓存读取一组手续费率。"""

    cache = load_fee_cache(path)
    values: dict[Any, float] = {}
    missing: list[str] = []
    for inst in columns:
        key = (str(date), str(inst))
        if key not in cache.index:
            missing.append(str(inst))
            continue
        values[inst] = _require_finite_nonnegative_scalar(cache.loc[key], f"fee_rate {date} {inst}")
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"fee_cache missing {len(missing)} instruments for {date}: {preview}")
    return pd.Series(values)


def read_fee_rate(date: str, inst: str, fee_cache: str | Path | None = None) -> float:
    """读取单合约手续费率。配置 fee_cache 时从本地缓存读取，否则读 metadata。"""

    if fee_cache:
        return float(read_fee_rates_from_cache(Path(fee_cache), date, pd.Index([inst])).loc[inst])
    
    # require_supported_runtime()
    exchange = get_env_exchange()
    
    if exchange == CF_EXCHANGE:
        return _require_finite_nonnegative_scalar(metadata.fee_rate(date, inst), f"fee_rate {date} {inst}")
    elif exchange == OKX_EXCHANGE:
        raise ValueError("read_fee_rate without fee_cache only supports CF 5min futures")
    else:
        raise ValueError(f"unsupported runtime EXCHANGE={exchange!r}")


def read_fee_rates(date: str, columns: pd.Index, fee_cache: str | Path | None = None) -> pd.Series:
    """读取一组合约手续费率。配置 fee_cache 时不访问 metadata Postgres。"""

    if len(columns) == 0:
        return pd.Series(dtype=float)
    if fee_cache:
        return read_fee_rates_from_cache(Path(fee_cache), date, columns)
    return pd.Series({inst: read_fee_rate(date, str(inst)) for inst in columns})


## add for legacy bn 5min

from mona import sdk_offline

def read_fee(date, insts) -> pd.Series:
    exchange = get_env_exchange()
    assert exchange in ["CF", "CF10s", "CF5m"]
    # return pd.Series([sdk_offline.read_fee_closehistory("CF", inst, date) for inst in insts], index=insts)
    return pd.Series([sdk_offline.read_fee("CF", inst, date) for inst in insts], index=insts)


def read_basedata(date, inst) -> pd.DataFrame:
    exchange = get_env_exchange()
    return sdk_offline.read_basedata(exchange, date, inst)    
    # return template(
    #     {
    #         # "binance": crypto_v1.read_basedata,
    #         "other": sdk_offline.read_basedata,
    #         # "bybit": sdk_offline.read_basedata,
    #         # "CF": sdk_offline.read_basedata,
    #         # "coinbase": sdk_offline.read_basedata,
    #     },
    #     date,
    #     inst,
    # )

def read_orderbook(date, inst) -> pd.DataFrame:
    exchange = get_env_exchange()
    return sdk_offline.read_orderbook(exchange, date, inst)
    # return template(
    #     {
    #         # "binance": crypto_v1.read_orderbook,
    #         "other": ,
    #         # "bybit": sdk_offline.read_orderbook,
    #         # "CF": sdk_offline.read_orderbook,
    #         # "coinbase": sdk_offline.read_orderbook,
    #     },
    #     date,
    #     inst,
    # )
