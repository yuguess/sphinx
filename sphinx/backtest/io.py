import os
import pandas as pd
from typing import Any, Callable

from sphinx.util.runtime_config import CF_EXCHANGE, OKX_EXCHANGE
from sphinx.util.exchange_api import get_dates, read_universe, get_index, read_holding, read_market_data, read_fee_rates
from .validation import require_quantity_frame, Frame, require_index_equal, require_no_nan, require_nonnegative_number
from .validation import require_mapping, require_finite_nonnegative_scalar
from .config import previous_selected_date


# def load_exchange_api() -> Any:
#     """延迟导入 exchange_api。

#     exchange_api 会导入 f2，所以必须等 `configure_environment()` 之后再导入。
#     """

#     module_name = "_agent_cta_exchange_api"
#     if module_name in sys.modules:
#         return sys.modules[module_name]
#     spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "exchange_api.py")
#     if spec is None or spec.loader is None:
#         raise SystemExit("failed to load root exchange_api.py")
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module
#     spec.loader.exec_module(module)
#     return module


def read_valid_frame(date: str, columns: pd.Index) -> Frame:
    exchange = os.environ.get("EXCHANGE", "")
    if exchange == CF_EXCHANGE:
        return read_market_frame(date, columns, "feature/valid").eq(1)
    elif exchange == OKX_EXCHANGE:
        return read_market_frame(date, columns, "feature/ret1m").abs().fillna(0).gt(-1)
    else:
        raise SystemExit(f"only {CF_EXCHANGE}/{OKX_EXCHANGE} valid path is supported, got {exchange!r}")


def require_ep_holding_axes_match(ep_holding: Frame, universe: pd.Series, expected_index: pd.Index, date: str, ep_holding_univ_name: str) -> None:
    """holding 的列和时间轴必须严格等于 universe 和标准 bar index。"""

    universe_columns = pd.Index(universe.index)
    if universe_columns.has_duplicates:
        raise ValueError(f"universe {date} {ep_holding_univ_name} has duplicate instruments")
    if expected_index.has_duplicates:
        raise ValueError(f"get_index {date} has duplicate timestamps")
    require_index_equal(ep_holding.columns, universe_columns, f"ep_holding {date} columns must match universe {ep_holding_univ_name}")
    require_index_equal(ep_holding.index, expected_index, f"ep_holding {date} index must match get_index(date)")


def read_ep_holding_quantity(date: str, ep_holding_univ_name: str, ep_holding_key: str) -> Frame:
    """从 DB holding 读取 EP 数量仓位。

    EP holding 是第 t 行的目标 EP 仓位，经过一个 freq 的 TWAP 执行到达。
    这里读到的必须已经是“合约数量”，不是 pct 仓位。
    """

    ep_holding = read_holding(date, ep_holding_univ_name, ep_holding_key)
    require_quantity_frame(ep_holding, f"ep_holding {date}")
    universe = read_universe(date, ep_holding_univ_name)
    expected_index = get_index(date)
    require_ep_holding_axes_match(ep_holding, universe, expected_index, date, ep_holding_univ_name)
    return ep_holding


def selected_dates(config: dict[str, Any]) -> list[str]:
    return get_dates(config["start_date"], config["end_date"])


def append_next_available_date(dates: list[str]) -> list[str]:
    """回测最后一天的最后一根交易需要下一交易日第一根执行数据。"""

    if not dates:
        return []
    elif dates:
        last_date = dates[-1]
        search_end = (pd.Timestamp(last_date) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        try:
            future_dates = [date for date in get_dates(last_date, search_end) if date > last_date]
        except Exception:
            return list(dates)
        if future_dates:
            return [*dates, future_dates[0]]
        elif not future_dates:
            return list(dates)
        else:
            raise ValueError(f"unexpected future dates state after {last_date}")
    else:
        raise ValueError("unexpected dates state")


def read_close(date: str, columns: pd.Index) -> Frame:
    """按合约列读取 close，并拼成宽表。"""
    return read_market_frame(date, columns, "kline/close")


def read_market_frame(date: str, columns: pd.Index, data_name: str) -> Frame:
    """把 exchange_api 的单合约 Series 拼成回测需要的宽表。

    数据路径只在 exchange_api 里定义；这里不再知道 f2 的底层路径。
    """

    if len(columns) == 0:
        return pd.DataFrame()
    items = [read_market_data(date, inst, data_name).rename(inst) for inst in columns]
    return pd.concat(items, axis=1)


def read_ep_holding_and_close_with_history_from_reader(
    dates: list[str], date: str, read_quantity: Callable[[str], Frame]) -> tuple[Frame, Frame, pd.Index]:
    """从统一 holding reader 读取当天 EP 数量仓位和 close，并拼上上一个交易日最后一行。

    为什么要拼历史行：
    1. `pre_cost_pnl` 用 `ep_holding.shift(1)` 和 `close.diff()`。
    2. `qty_trade` 用 `ep_holding.diff()`。

    这两个公式在当天第一根 bar 上都需要看到“上一根”。
    所以我们显式拼一行历史数据，最后再用 `.iloc[1:]` 丢掉历史行。
    """

    ep_holding_today = read_quantity(date)
    require_quantity_frame(ep_holding_today, f"ep_holding {date}")
    today_columns = ep_holding_today.columns

    if date == dates[0]:
        columns = today_columns
        forced_exit_columns = pd.Index([])
        close_today = read_close(date, columns).reindex(index=ep_holding_today.index, columns=columns)

        # 回测第一天没有更早的本策略仓位。用 0 仓位补一行。
        # 收盘价用当天第一根收盘价，保证第一根价格变化为 0。
        # 这里不要对当天真实收盘价做向后填充，否则会掩盖数据缺失。
        previous_ep_holding = pd.DataFrame(0.0, columns=columns, index=ep_holding_today.index[:1] - pd.Timedelta("1ns"))
        previous_close = pd.DataFrame([close_today.iloc[0]], columns=columns, index=previous_ep_holding.index)
    elif date != dates[0]:
        # 这里必须用已选交易日列表的前一天，不能用自然日。
        # 否则周一会去读周日文件，触发 KeyError。
        previous_date = previous_selected_date(dates, date)
        previous_ep_holding_raw = read_quantity(previous_date)
        require_quantity_frame(previous_ep_holding_raw, f"ep_holding {previous_date}")

        # 用前后两天列的并集。
        # 如果某个合约今天被 universe 移除，今天数量会补成 0，
        # 这会显式形成一笔清仓交易，而不是把昨天的仓位悄悄丢掉。
        columns = previous_ep_holding_raw.columns.union(today_columns)
        ep_holding_today = ep_holding_today.reindex(columns=columns, fill_value=0.0)
        forced_exit_columns = previous_ep_holding_raw.columns.difference(today_columns)
        previous_ep_holding = previous_ep_holding_raw.iloc[-1:]
        previous_ep_holding = previous_ep_holding.reindex(columns=columns, fill_value=0.0)

        previous_close_raw = read_close(previous_date, columns).reindex(columns=columns)
        if len(previous_close_raw.index) == 0:
            raise ValueError(f"close {previous_date} must have at least one row")
        previous_close_last = previous_close_raw.iloc[-1]
        close_today = read_close(date, today_columns).reindex(columns=columns)
        for column in forced_exit_columns:
            try:
                removed_close_today = read_close(date, pd.Index([column]))
                if len(removed_close_today.index) == 0:
                    raise ValueError(f"close {date} {column} must have at least one row")
                close_today.loc[:, column] = float(removed_close_today.iloc[0, 0])
            except KeyError:
                close_today.loc[:, column] = float(previous_close_last.loc[column])
        close_today = close_today.reindex(index=ep_holding_today.index, columns=columns)
        previous_close = pd.DataFrame([previous_close_last.to_numpy()], index=previous_ep_holding.index, columns=columns)

    ep_holding = pd.concat([previous_ep_holding, ep_holding_today], axis=0)
    require_no_nan(ep_holding, f"ep_holding with history {date}")
    close = pd.concat([previous_close, close_today], axis=0)
    return ep_holding, close, forced_exit_columns


def read_market_frame_zeroing_columns(
    date: str, columns: pd.Index, data_name: str, index: pd.Index | None = None, zero_columns: pd.Index | None = None) -> Frame:

    """读取 market frame，并把指定列补 0 且不访问这些列的数据源。"""

    zero_columns = zero_columns if zero_columns is not None else pd.Index([])
    zero = zero_columns.intersection(columns)
    present = columns.difference(zero)
    if len(columns) == 0:
        return pd.DataFrame(index=index, columns=columns)
    elif len(zero) == 0:
        frame = read_market_frame(date, columns, data_name)
    elif len(present) == 0:
        frame_index = index if index is not None else pd.RangeIndex(1)
        frame = pd.DataFrame(0.0, index=frame_index, columns=columns)
    else:
        frame = read_market_frame(date, present, data_name)

    if index is not None:
        frame = frame.reindex(index=index, columns=columns)
    else:
        frame = frame.reindex(columns=columns)
    if len(zero) > 0:
        frame.loc[:, zero] = 0.0
    return frame


def read_market_frame_zeroing_columns(
    date: str, columns: pd.Index, data_name: str,
    index: pd.Index | None = None, zero_columns: pd.Index | None = None) -> Frame:
    """读取 market frame，并把指定列补 0 且不访问这些列的数据源。"""

    zero_columns = zero_columns if zero_columns is not None else pd.Index([])
    zero = zero_columns.intersection(columns)
    present = columns.difference(zero)
    if len(columns) == 0:
        return pd.DataFrame(index=index, columns=columns)
    elif len(zero) == 0:
        frame = read_market_frame(date, columns, data_name)
    elif len(present) == 0:
        frame_index = index if index is not None else pd.RangeIndex(1)
        frame = pd.DataFrame(0.0, index=frame_index, columns=columns)
    else:
        frame = read_market_frame(date, present, data_name)

    if index is not None:
        frame = frame.reindex(index=index, columns=columns)
    else:
        frame = frame.reindex(columns=columns)
    if len(zero) > 0:
        frame.loc[:, zero] = 0.0
    return frame


def fee_for_date(config: dict[str, Any], date: str, columns: pd.Index) -> float | pd.Series:
    """读取手续费。

    如果配置了 `backtest.fee_cache`，exchange_api 会优先从本地缓存读取。
    """

    exchange = os.environ.get("EXCHANGE", "")
    if len(columns) == 0:
        return pd.Series(dtype=float)
    if exchange == OKX_EXCHANGE:
        fee = require_nonnegative_number(config.get("fee"), "fee")
        return pd.Series(fee, index=columns)
    elif exchange == CF_EXCHANGE:
        fee_cache = require_mapping(config.get("backtest", {}), "backtest").get("fee_cache")
        fee_rate = read_fee_rates(date, columns, fee_cache=fee_cache)
        if not isinstance(fee_rate, pd.Series):
            raise ValueError(f"fee_rate {date} must be a Series")
        missing = columns.difference(fee_rate.index)
        if len(missing) > 0:
            raise ValueError(f"fee_rate missing active instruments: {list(missing)}")
        return pd.Series(
            {
                inst: require_finite_nonnegative_scalar(fee_rate.loc[inst], f"fee_rate {date} {inst}")
                for inst in columns
            }
        )
    else:
        raise SystemExit(f"only {CF_EXCHANGE}/{OKX_EXCHANGE} fee path is supported, got {exchange!r}")


def funding_fee_for_date(
    date: str, columns: pd.Index, index: pd.Index, zero_missing_columns: pd.Index | None = None) -> Frame:
    """读取逐 bar 资金费率。OKX 用真实结算资金费，CF 没有资金费。"""

    exchange = os.environ.get("EXCHANGE", "")
    if exchange == OKX_EXCHANGE:
        funding_fee = read_market_frame_zeroing_columns(
            date,
            columns,
            "funding/settle",
            index=index,
            zero_columns=zero_missing_columns,
        )
        funding_fee = funding_fee.fillna(0.0)
        require_no_nan(funding_fee, f"funding_fee {date}")
        return funding_fee
    elif exchange == CF_EXCHANGE:
        return pd.DataFrame(0.0, index=index, columns=columns)
    else:
        raise SystemExit(f"only {CF_EXCHANGE}/{OKX_EXCHANGE} funding fee path is supported, got {exchange!r}")
