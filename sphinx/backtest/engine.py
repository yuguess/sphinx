import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from multiprocessing import Pool

from sphinx.util.runtime_config import get_env_exchange
from .results import QuantityResult, build_result
from .config import OKX_EXCHANGE, CF_EXCHANGE, configured_capacity_violation_tolerance
from .config import split_dates_for_jobs, worker_count_for_chunks, next_selected_date, require_make_execution, should_repair_invalid_bar_holding, CAPACITY_VIOLATION_TOLERANCE
from .io import append_next_available_date, read_ep_holding_quantity, read_ep_holding_and_close_with_history_from_reader, read_valid_frame, read_market_frame_zeroing_columns
from .io import funding_fee_for_date, fee_for_date
from .validation import require_no_nan, Frame, require_quantity_frame, require_positive_number, require_positive_data, zero_nan_for_inactive_position
from .validation import invalid_number_mask, first_bad_location, require_data, require_nonnegative_data, require_finite_nonnegative_scalar


CF_CONTRACT_PATTERN = re.compile(r"^([A-Z]+)(\d{4})$")


@dataclass
class ExecutionData:
    """make 需要的逐 bar 数据。
    这里尽量只存原始数据，不提前算 limit。
    limit 在 `compute_make_costs()` 里统一计算，方便检查口径。
    """
    # 柱子的收盘价。仓位约束用这个价格。
    close_price: Frame
    # 柱子的总成交额。仓位约束用它的一个固定比例。
    bar_turnover: Frame


@dataclass
class BarPositionFlow:
    """一个交易日里每个 bar 的持仓收益和交易数量。"""

    # 只考虑价格变化，不扣任何交易成本。
    pre_cost_pnl: Frame
    # 当前 bar 要执行的数量变化。
    qty_trade: Frame
    # 时间加权平均价格滑点的参考价格：上一根收盘价。
    twap_reference_price: Frame


@dataclass
class CostBreakdown:
    """把总成本拆成互相独立、可检查的部分。"""
    # 手续费成本。
    fee_cost: Frame
    # 下一根执行 bar 的半价差成本。
    spread_cost: Frame
    # 一秒时间加权平均价格相对上一根收盘价的价格滑点。
    twap_cost: Frame
    # 所有成本之和。
    total_cost: Frame
    # 成交额 / nav，回测里的换手率口径。
    turnover: Frame
    # 当前 bar 最多允许交易的金额 / nav，约束检查口径。
    limit_pct: Frame
    # 当前交易按 close 计价的金额 / nav，约束检查口径。
    trade_pct: Frame
    # 超过约束的金额 / nav。正确情况下只能很小。
    over_pct: Frame
    # 当前 bar 允许买入的总数量容量，按 close 口径换算。
    buy_limit_qty: Frame
    # 当前 bar 允许卖出的总数量容量，按 close 口径换算。
    sell_limit_qty: Frame
    # 买入数量超过当前 bar 约束的部分，按 close 口径换算。
    buy_over_qty: Frame
    # 卖出数量超过当前 bar 约束的部分，按 close 口径换算。
    sell_over_qty: Frame


def repair_ep_holding_on_invalid_bar(date: str, ep_holding: Frame, forced_exit_columns: pd.Index | None = None) -> Frame:
    """把 invalid bar 上的目标仓位延续上一行，仅用于显式开启的并行分段回测。"""

    current_index = ep_holding.index[1:]
    forced_exit_columns = forced_exit_columns if forced_exit_columns is not None else pd.Index([])
    valid_columns = ep_holding.columns[~ep_holding.columns.isin(forced_exit_columns)]
    valid = pd.DataFrame(True, index=current_index, columns=ep_holding.columns)
    if len(valid_columns) > 0:
        valid.loc[:, valid_columns] = read_valid_frame(date, valid_columns).reindex(index=current_index, columns=valid_columns).eq(1)
    repaired = ep_holding.copy()
    previous = repaired.iloc[0].copy()
    for timestamp in current_index:
        row = repaired.loc[timestamp].copy()
        keep_previous = ~valid.loc[timestamp]
        row = row.where(~keep_previous, previous)
        repaired.loc[timestamp] = row
        previous = row
    require_quantity_frame(repaired, f"repaired_ep_holding {date}")
    return repaired


def compute_position_flow(ep_holding: Frame, close: Frame, nav: float) -> BarPositionFlow:
    """把 EP 数量仓位转换成逐 bar 收益和逐 bar 交易数量。

    这里是 native 数量路径的核心。

    生成器在第 t 行先把旧 EP 仓位按 `ret[t]` 漂移到当前 close，
    再用第 t 行的成交量约束求出新目标 EP 仓位，并把它写成 `ep_holding[t]`。
    第 t 行的 EP holding 变化在下一根 TWAP 执行，所以价格滑点要用
    `twap_slippage[t+1]`，其参考价是 `close[t]`。

    因此在当前 bar t 上：
    - 持仓收益来自上一行 EP 数量：`ep_holding[t-1] * (close[t] - close[t-1]) / nav`
    - 当前生成的交易来自当前目标变化：`ep_holding[t] - ep_holding[t-1]`
    """

    require_positive_number(nav, "nav")
    require_no_nan(ep_holding, "ep_holding")
    close_needed = (ep_holding.abs() > 0.0) | (ep_holding.shift(1).fillna(0.0).abs() > 0.0)
    require_positive_data(close, close_needed, "close")

    price_change = close.diff()
    previous_ep_holding = ep_holding.shift(1)
    pre_cost_pnl = previous_ep_holding * price_change / nav
    pre_cost_pnl = zero_nan_for_inactive_position(pre_cost_pnl, previous_ep_holding, "pre_cost_pnl")

    # 当前行目标 - 上一行目标，就是当前行信号触发的交易数量。
    qty_trade = ep_holding.diff()

    # 数据字段 feature/twap_slippage[t+1] = twap_1s[t+1] / close[t] - 1。
    # 所以第 t 行交易要用第 t 行 close 作为下一根 TWAP 的参考价。
    twap_reference_price = close

    # 第一行是为了 shift/diff 拼进去的历史行，不属于当天输出。
    return BarPositionFlow(pre_cost_pnl=pre_cost_pnl.iloc[1:], qty_trade=qty_trade.iloc[1:], twap_reference_price=twap_reference_price.iloc[1:])


def validate_ep_holding_changes_on_valid_bar(date: str, qty_trade: Frame, forced_exit_columns: pd.Index | None = None) -> None:
    """EP holding 主动变化只能发生在 valid > 0 的 bar。

    强制清仓列不是策略主动交易，不参与本校验。
    """

    require_no_nan(qty_trade, "qty_trade")
    forced_exit_columns = forced_exit_columns if forced_exit_columns is not None else pd.Index([])
    valid_columns = qty_trade.columns[~qty_trade.columns.isin(forced_exit_columns)]
    used = qty_trade.reindex(columns=valid_columns).abs() > 0.0
    if not used.any().any():
        return

    active_columns = valid_columns[used.any(axis=0).to_numpy()]
    valid = read_valid_frame(date, active_columns).reindex(
        index=qty_trade.index,
        columns=active_columns,
    )
    used_active = used.reindex(columns=active_columns, fill_value=False)
    invalid_or_missing = (invalid_number_mask(valid) | valid.le(0.0)) & used_active
    if invalid_or_missing.any().any():
        timestamp, instrument = first_bad_location(invalid_or_missing)
        raise ValueError(f"ep_holding changes on invalid bar at {timestamp} {instrument}")


def fallback_execution_feature_row(current: Frame, columns: pd.Index, next_columns: pd.Index, data_name: str) -> Frame:
    if len(current.index) == 0:
        raise ValueError(f"{data_name} current frame has no rows for fallback")
    fallback_index = pd.Index([current.index[-1] + pd.Timedelta("1ns")])
    fallback = pd.DataFrame(np.nan, index=fallback_index, columns=columns)
    if data_name == "feature/twap_slippage":
        fallback.loc[:, next_columns] = 0.0
    elif data_name == "feature/half_spread_mean":
        value = current.ffill().iloc[-1].reindex(next_columns)
        value = value.where(np.isfinite(value) & value.ge(0.0), 0.0)
        fallback.loc[fallback_index[0], next_columns] = value
    else:
        raise ValueError(f"unsupported execution feature fallback: {data_name}")
    return fallback


def read_execution_feature_frame(
    date: str, next_date: str | None, columns: pd.Index, index: pd.Index,
    next_columns: pd.Index,
    data_name: str,
    allow_fallback: bool = False,
    zero_missing_columns: pd.Index | None = None,
    zero_missing_next_columns: pd.Index | None = None,
) -> Frame:
    """读取当前日执行特征，并在需要时拼入下一交易日第一根。"""

    zero_missing_columns = zero_missing_columns if zero_missing_columns is not None else pd.Index([])
    zero_missing_next_columns = zero_missing_next_columns if zero_missing_next_columns is not None else pd.Index([])
    current = read_market_frame_zeroing_columns(date, columns, data_name, index=index, zero_columns=zero_missing_columns)
    if len(next_columns) == 0:
        return current
    elif next_date is None:
        if allow_fallback:
            next_row = fallback_execution_feature_row(current, columns, next_columns, data_name)
            return pd.concat([current, next_row], axis=0)
        elif not allow_fallback:
            raise ValueError(f"{data_name} next trading date is required for last-bar execution on {date}")
        else:
            raise ValueError(f"unexpected fallback state: {allow_fallback!r}")
    elif next_date is not None:
        try:
            next_frame = read_market_frame_zeroing_columns(next_date, next_columns, data_name, zero_columns=zero_missing_next_columns)
        except Exception as exc:
            if allow_fallback:
                next_row = fallback_execution_feature_row(current, columns, next_columns, data_name)
                return pd.concat([current, next_row], axis=0)
            elif not allow_fallback:
                raise ValueError(f"{data_name} {next_date} is required for last-bar execution on {date}") from exc
            else:
                raise ValueError(f"unexpected fallback state: {allow_fallback!r}") from exc
        if len(next_frame.index) == 0:
            if allow_fallback:
                next_row = fallback_execution_feature_row(current, columns, next_columns, data_name)
                return pd.concat([current, next_row], axis=0)
            elif not allow_fallback:
                raise ValueError(f"{data_name} {next_date} has no rows for last-bar execution on {date}")
            else:
                raise ValueError(f"unexpected fallback state: {allow_fallback!r}")
        next_row = next_frame.iloc[:1].reindex(columns=columns)
        if allow_fallback:
            fallback_row = fallback_execution_feature_row(current, columns, next_columns, data_name)
            missing = invalid_number_mask(next_row.reindex(columns=next_columns))
            if missing.any().any():
                next_row.loc[:, next_columns] = next_row.loc[:, next_columns].mask(missing, fallback_row.loc[:, next_columns].to_numpy())
        return pd.concat([current, next_row], axis=0)
    else:
        raise ValueError(f"unexpected next_date state: {next_date!r}")


def next_bar_data(value: Frame, index: pd.Index, columns: pd.Index) -> Frame:
    """把当前 bar 对齐到下一根执行 bar，保留附加的跨日第一根。"""

    extra_index = pd.Index([item for item in value.index if item not in index])
    if len(extra_index) > 0:
        aligned_index = pd.Index(index).append(extra_index)
    elif len(extra_index) == 0:
        aligned_index = pd.Index(index)
    else:
        raise ValueError("unexpected extra execution index state")
    aligned = value.reindex(index=aligned_index, columns=columns)
    return aligned.shift(-1).reindex(index=index, columns=columns)


def fee_rate_for_cost(fee_rate: float | pd.Series, columns: pd.Index, active_columns: pd.Index) -> pd.Series:
    """把手续费整理成全列 Series，只要求有交易的列必须有有效费率。"""

    if isinstance(fee_rate, pd.Series):
        missing = active_columns.difference(fee_rate.index)
        if len(missing) > 0:
            raise ValueError(f"fee_rate missing active instruments: {list(missing)}")
        aligned = fee_rate.reindex(columns)
        for instrument in active_columns:
            aligned.loc[instrument] = require_finite_nonnegative_scalar(
                aligned.loc[instrument],
                f"fee_rate {instrument}",
            )
        return aligned.fillna(0.0)

    scalar = require_finite_nonnegative_scalar(fee_rate, "fee_rate")
    return pd.Series(scalar, index=columns)


def compute_make_costs(
    qty_trade: Frame, execution: ExecutionData, fee_rate: float | pd.Series, twap_reference_price: Frame, twap_slippage: Frame,
    half_spread_mean: Frame, turnover_limit_rate: float, nav: float, capacity_violation_tolerance: float = CAPACITY_VIOLATION_TOLERANCE,
    forced_exit_columns: pd.Index | None = None) -> CostBreakdown:

    """计算 make 模式下的成本

    输入 `qty_trade` 是数量，不是 pct 仓位。
    正数表示买入，负数表示卖出。

    make 不加固定价差惩罚，但使用数据里的 `half_spread_mean`，并且和 TWAP 一样使用下一根执行 bar。
    容量检查、换手和手续费都用 close 口径。

    TWAP 成本的时间口径也很重要：
    - cncf 的 `feature/twap_slippage[i]` 定义为 `twap_1s[i] / close[i-1] - 1`。
    - native 数量路径里 `qty_trade[i] = quantity[i] - quantity[i-1]` 表示第 i 行信号
      触发的交易。
    - 这笔交易用下一根 TWAP 执行，所以成本使用 `twap_slippage[i+1]`，并用
      `close[i]` 作为名义成交额参考价。
    - 容量约束仍然使用当前 bar 的成交额，因为策略生成仓位时只能使用当前可见容量。
    """

    require_positive_number(nav, "nav")
    require_positive_number(turnover_limit_rate, "turnover_limit_rate")
    require_no_nan(qty_trade, "qty_trade")
    if half_spread_mean is None:
        raise ValueError("half_spread_mean is required")

    buy_qty = qty_trade.where(qty_trade > 0, 0.0)
    sell_qty = (-qty_trade).where(qty_trade < 0, 0.0)
    used = qty_trade.abs() > 0.0
    forced_exit_columns = forced_exit_columns if forced_exit_columns is not None else pd.Index([])
    constrained = used.copy()
    constrained.loc[:, forced_exit_columns.intersection(qty_trade.columns)] = False
    active_columns = qty_trade.columns[used.any(axis=0).to_numpy()]

    execution_twap_slippage = next_bar_data(twap_slippage, qty_trade.index, qty_trade.columns)
    execution_half_spread = next_bar_data(half_spread_mean, qty_trade.index, qty_trade.columns)
    require_positive_data(execution.close_price, used, "close_price")
    require_nonnegative_data(execution.bar_turnover, constrained, "bar_turnover")
    require_positive_data(twap_reference_price, used, "twap_reference_price")
    require_data(execution_twap_slippage, used, "twap_slippage")
    require_nonnegative_data(execution_half_spread, used, "half_spread_mean")

    # 约束检查：
    # 原组合生成器在 pct 口径下约束交易额。
    # 对 make 来说，long_price * long_volume = bar_turnover。
    # 容量约束保持当前 bar 口径，因为仓位生成时只能使用当前可见容量。
    # 所以 limit 实际是 bar_turnover * turnover_limit_rate / nav。
    buy_trade_pct = (buy_qty * execution.close_price / nav).where(buy_qty > 0.0, 0.0)
    sell_trade_pct = (sell_qty * execution.close_price / nav).where(sell_qty > 0.0, 0.0)
    trade_pct = buy_trade_pct + sell_trade_pct
    limit_pct = (execution.bar_turnover * turnover_limit_rate / nav).where(constrained, 0.0)

    buy_over_pct = (buy_trade_pct - limit_pct).where(
        constrained & (buy_trade_pct > limit_pct),
        0.0,
    )
    sell_over_pct = (sell_trade_pct - limit_pct).where(
        constrained & (sell_trade_pct > limit_pct),
        0.0,
    )
    over_pct = buy_over_pct + sell_over_pct

    max_over_pct = over_pct.max().max()
    if max_over_pct > capacity_violation_tolerance:
        timestamp, instrument = over_pct.stack(future_stack=True).idxmax()
        raise ValueError("capacity violation: "
                         f"max over pct {max_over_pct:.6g} at {timestamp} {instrument}")

    limit_qty = (limit_pct * nav / execution.close_price).where(used, 0.0)
    buy_over_qty = (buy_over_pct * nav / execution.close_price).where(buy_over_pct > 0.0, 0.0)
    sell_over_qty = (sell_over_pct * nav / execution.close_price).where(sell_over_pct > 0.0, 0.0)

    # 无交易的位置必须显式写成 0。
    # 如果这里直接做 `0 * NaN`，结果仍然是 NaN；
    # 后面 pandas 汇总时又可能把 NaN 静默跳过，掩盖坏数据。
    buy_turnover = (buy_qty * execution.close_price / nav).where(buy_qty > 0.0, 0.0)
    sell_turnover = (sell_qty * execution.close_price / nav).where(sell_qty > 0.0, 0.0)
    turnover = buy_turnover + sell_turnover

    fee_rate_series = fee_rate_for_cost(fee_rate, qty_trade.columns, active_columns)
    fee_cost = turnover.mul(fee_rate_series, axis=1).where(used, 0.0)

    # make 半价差成本：
    # 第 t 行信号触发的交易在 t+1 执行，所以使用 half_spread_mean[t+1]。
    # 买入按 ask 多付半价差，卖出按 bid 少收半价差，成本均为正。
    spread_cost = qty_trade.abs() * execution_half_spread / nav
    spread_cost = zero_nan_for_inactive_position(spread_cost, qty_trade, "spread_cost")

    # 时间加权平均价格成本：
    # qty_trade[t] * close[t] / nav 是按信号行收盘价计价的交易额占净值比例。
    # twap_slippage[t+1] 是下一根 TWAP 相对 close[t] 的涨跌幅。
    twap_cost = qty_trade * twap_reference_price / nav * execution_twap_slippage
    twap_cost = zero_nan_for_inactive_position(twap_cost, qty_trade, "twap_cost")

    total_cost = fee_cost + spread_cost + twap_cost
    require_no_nan(limit_pct, "limit_pct")
    require_no_nan(trade_pct, "trade_pct")
    require_no_nan(over_pct, "over_pct")
    require_no_nan(limit_qty, "limit_qty")
    require_no_nan(buy_over_qty, "buy_over_qty")
    require_no_nan(sell_over_qty, "sell_over_qty")
    require_no_nan(turnover, "turnover")
    require_no_nan(fee_cost, "fee_cost")
    require_no_nan(spread_cost, "spread_cost")
    require_no_nan(twap_cost, "twap_cost")
    require_no_nan(total_cost, "total_cost")
    return CostBreakdown(
        fee_cost=fee_cost,
        spread_cost=spread_cost,
        twap_cost=twap_cost,
        total_cost=total_cost,
        turnover=turnover,
        limit_pct=limit_pct,
        trade_pct=trade_pct,
        over_pct=over_pct,
        buy_limit_qty=limit_qty,
        sell_limit_qty=limit_qty,
        buy_over_qty=buy_over_qty,
        sell_over_qty=sell_over_qty,
    )


def compute_ep_holding_value_ratio(ep_holding: Frame, close: Frame, nav: float) -> pd.Series:
    """计算每根柱子的持仓价值 / NAV。

    口径是所有合约的名义持仓价值绝对值之和：
    `sum(abs(ep_holding * close)) / nav`。
    零持仓位置可以没有价格；非零持仓位置必须有有效正价格。
    """

    require_positive_number(nav, "nav")
    require_no_nan(ep_holding, "ep_holding")
    active = ep_holding.abs() > 0.0
    require_positive_data(close, active, "ep_holding_close")

    ep_holding_value = (ep_holding.abs() * close).where(active, 0.0)
    require_no_nan(ep_holding_value, "ep_holding_value")
    ratio = ep_holding_value.sum(axis=1) / nav
    if not np.isfinite(ratio.to_numpy()).all():
        raise ValueError("ep_holding_value_ratio has non-finite value")
    return ratio.rename("ep_holding_value_ratio")


def read_make_execution_data(date: str, columns: pd.Index, index: pd.Index, close: Frame, zero_turnover_columns: pd.Index | None = None) -> ExecutionData:
    """读取 make 模式的价格和成交额。
    make 不再额外加固定价差惩罚，但会使用数据里的 half_spread_mean。
    成交额、手续费、容量检查都用 close 口径。
    """

    close = close.reindex(index=index, columns=columns)
    zero_turnover_columns = zero_turnover_columns if zero_turnover_columns is not None else pd.Index([])
    market_turnover = read_market_frame_zeroing_columns(date, columns, "kline/turnover", index=index, zero_columns=zero_turnover_columns)

    return ExecutionData(
        close_price=close,
        bar_turnover=market_turnover,
    )


def compute_funding_fee_cost(ep_holding: Frame, close: Frame, funding_fee: Frame, nav: float) -> Frame:
    """资金费成本：正资金费扣多头、补空头。"""
    require_positive_number(nav, "nav")
    require_no_nan(ep_holding, "ep_holding")
    funding_fee = funding_fee.reindex(index=ep_holding.index, columns=ep_holding.columns).fillna(0.0)
    require_no_nan(funding_fee, "funding_fee")
    active = ep_holding.abs() > 0.0
    require_positive_data(close, active, "funding_fee_close")
    cost = ep_holding * close / nav * funding_fee
    return zero_nan_for_inactive_position(cost, ep_holding, "funding_fee_cost")

def write_diagnostics(diagnostics_dir: Path, date: str, flow: BarPositionFlow, costs: CostBreakdown, funding_fee_cost: Frame, total_cost: Frame, net_pnl: Frame) -> None:
    """输出逐 bar 诊断表，方便查某一天差异到底来自哪里。

    诊断数据通常很大，所以这里用 parquet，而不是 CSV。
    parquet 读写更快、体积更小，也能保留时间戳和数值类型。
    """

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    active_limit_qty = costs.buy_limit_qty.where(
        flow.qty_trade > 0,
        costs.sell_limit_qty.where(flow.qty_trade < 0, 0.0),
    )
    active_over_qty = costs.buy_over_qty.where(
        flow.qty_trade > 0,
        costs.sell_over_qty.where(flow.qty_trade < 0, 0.0),
    )
    items = {
        "qty_trade": flow.qty_trade,
        "twap_reference_price": flow.twap_reference_price,
        "limit_qty": active_limit_qty,
        "over_qty": active_over_qty,
        "trade_pct": costs.trade_pct,
        "limit_pct": costs.limit_pct,
        "over_pct": costs.over_pct,
        "pre_cost_pnl": flow.pre_cost_pnl,
        "fee_cost": costs.fee_cost,
        "spread_cost": costs.spread_cost,
        "twap_cost": costs.twap_cost,
        "funding_fee_cost": funding_fee_cost,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
        "turnover": costs.turnover,
    }
    stacked = pd.concat(
        [frame.stack(future_stack=True).rename(name) for name, frame in items.items()],
        axis=1,
    ).reset_index()
    stacked.columns = ["timestamp", "instrument", *items.keys()]
    stacked.insert(0, "date", date)
    stacked.to_parquet(
        diagnostics_dir / f"{date}.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )


def futures_product_from_contract(column: Any) -> str:
    """CF 合约列名必须是品种代码加四位年月，例如 AG2408。"""

    text = str(column)
    match = CF_CONTRACT_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid CF contract column {text!r}, expected format like AG2408")
    return match.group(1)


def sum_futures_contracts_by_product(frame: Frame) -> Frame:
    """把同一品种的多个合约列相加。"""

    renamed = frame.copy()
    renamed.columns = [futures_product_from_contract(column) for column in renamed.columns]
    return renamed.T.groupby(level=0, sort=False).sum().T


def strip_futures_contract_suffix_if_needed(pnl: Frame, turnover: Frame, cost: Frame) -> tuple[Frame, Frame, Frame]:
    """CF 数据列名通常是 AG2408 这种合约，输出时统一聚合成 AG。

    换月日可能同时出现 AG2408 和 AG2409。只改列名会得到重复列，
    所以这里必须按 product 再聚合一次。
    """

    exchange = get_env_exchange()
    if exchange == OKX_EXCHANGE:
        return pnl, turnover, cost
    elif exchange == CF_EXCHANGE:
        return (
            sum_futures_contracts_by_product(pnl),
            sum_futures_contracts_by_product(turnover),
            sum_futures_contracts_by_product(cost),
        )
    else:
        raise SystemExit("only runtime.EXCHANGE='CF' or 'okx' is supported")


def run_one_day_clear_from_history(
    config: dict[str, Any], date: str, next_date: str | None, allow_next_execution_fallback: bool, ep_holding: Frame, close: Frame, forced_exit_columns: pd.Index | None = None,
    zero_missing_next_execution_columns: pd.Index | None = None, diagnostics_dir: Path | None = None) -> tuple[Frame, Frame, Frame, pd.Series]:
    """对已拼好历史行的 EP holding 执行官方 PnL、成本、容量与 valid 检查。"""

    exec_info = require_make_execution(config)
    nav = float(config["nav"])
    columns = ep_holding.columns
    forced_exit_columns = forced_exit_columns if forced_exit_columns is not None else pd.Index([])
    zero_missing_next_execution_columns = zero_missing_next_execution_columns if zero_missing_next_execution_columns is not None else pd.Index([])
    if should_repair_invalid_bar_holding(config):
        ep_holding = repair_ep_holding_on_invalid_bar(date, ep_holding, forced_exit_columns)

    flow = compute_position_flow(ep_holding, close, nav)
    validate_ep_holding_changes_on_valid_bar(date, flow.qty_trade, forced_exit_columns)
    ep_holding_value_ratio = compute_ep_holding_value_ratio(ep_holding.iloc[1:], close.iloc[1:], nav)

    execution = read_make_execution_data(date, columns, flow.qty_trade.index, close.iloc[1:], zero_turnover_columns=forced_exit_columns)
    active_columns = columns[(flow.qty_trade.abs() > 0.0).any(axis=0).to_numpy()]
    fee_rate = fee_for_date(config, date, active_columns)
    if len(flow.qty_trade.index) > 0:
        next_execution_columns = pd.Index(flow.qty_trade.columns[flow.qty_trade.iloc[-1].abs().gt(0.0).to_numpy()])
    elif len(flow.qty_trade.index) == 0:
        next_execution_columns = pd.Index([])
    else:
        raise ValueError("unexpected empty flow state")

    twap_slippage = read_execution_feature_frame(date, next_date, columns, flow.qty_trade.index,
        next_execution_columns, "feature/twap_slippage", allow_fallback=allow_next_execution_fallback, zero_missing_columns=forced_exit_columns,
        zero_missing_next_columns=zero_missing_next_execution_columns,
    )

    half_spread_mean = read_execution_feature_frame(
        date, next_date, columns, flow.qty_trade.index, next_execution_columns, "feature/half_spread_mean",
        allow_fallback=allow_next_execution_fallback, zero_missing_columns=forced_exit_columns,
        zero_missing_next_columns=zero_missing_next_execution_columns,
    )

    funding_fee = funding_fee_for_date(date, columns, flow.pre_cost_pnl.index, zero_missing_columns=forced_exit_columns)

    costs = compute_make_costs(
        qty_trade=flow.qty_trade, execution=execution, fee_rate=fee_rate, twap_reference_price=flow.twap_reference_price, twap_slippage=twap_slippage,
        half_spread_mean=half_spread_mean, turnover_limit_rate=float(exec_info["turnover_limit_rate"]), nav=nav,
        forced_exit_columns=forced_exit_columns, capacity_violation_tolerance=configured_capacity_violation_tolerance(config))
    funding_fee_cost = compute_funding_fee_cost(ep_holding.iloc[1:], close.iloc[1:], funding_fee, nav)
    total_cost = costs.total_cost + funding_fee_cost

    pnl = flow.pre_cost_pnl - total_cost
    turnover = costs.turnover
    cost = total_cost
    require_no_nan(pnl, "pnl")
    require_no_nan(turnover, "turnover")
    require_no_nan(cost, "cost")

    if diagnostics_dir is not None:
        write_diagnostics(diagnostics_dir, date, flow, costs, funding_fee_cost, total_cost, pnl)

    pnl, turnover, cost = strip_futures_contract_suffix_if_needed(pnl, turnover, cost)
    return pnl, turnover, cost, ep_holding_value_ratio


def run_one_day_clear_from_reader(
    config: dict[str, Any],
    dates: list[str],
    execution_dates: list[str],
    last_requested_date: str,
    date: str,
    read_quantity: Callable[[str], Frame],
    read_next_columns: Callable[[str], pd.Index | None],
    diagnostics_dir: Path | None = None,
) -> tuple[Frame, Frame, Frame, pd.Series]:
    """用统一数量 holding reader 跑一个交易日。"""

    ep_holding, close, forced_exit_columns = read_ep_holding_and_close_with_history_from_reader(dates, date, read_quantity)
    next_date = next_selected_date(execution_dates, date)
    allow_next_execution_fallback = date == last_requested_date
    zero_missing_next_execution_columns = pd.Index([])
    if next_date is not None and not allow_next_execution_fallback:
        next_columns = read_next_columns(next_date)
        if next_columns is not None:
            zero_missing_next_execution_columns = ep_holding.columns.difference(next_columns)
    return run_one_day_clear_from_history(
        config, date, next_date, allow_next_execution_fallback,
        ep_holding, close, forced_exit_columns, zero_missing_next_execution_columns=zero_missing_next_execution_columns,
        diagnostics_dir=diagnostics_dir,
    )


def run_one_day_clear(
    config: dict[str, Any],
    dates: list[str],
    execution_dates: list[str],
    last_requested_date: str,
    date: str,
    ep_holding_univ_name: str,
    ep_holding_key: str,
    diagnostics_dir: Path | None = None,
) -> tuple[Frame, Frame, Frame, pd.Series]:
    """跑一个交易日。

    返回的三个 DataFrame 都是按分钟/5分钟 bar、按品种列展开，
    最后一个 Series 是每根柱子的 EP holding 价值 / NAV：
    - pnl: 扣完所有成本后的净收益率
    - turnover: 成交额 / nav
    - cost: 总成本
    - ep_holding_value_ratio: sum(abs(ep_holding * close)) / nav
    """

    def read_quantity(read_date: str) -> Frame:
        return read_ep_holding_quantity(read_date, ep_holding_univ_name, ep_holding_key)

    def read_next_columns(next_date: str) -> pd.Index:
        return read_ep_holding_quantity(next_date, ep_holding_univ_name, ep_holding_key).columns

    return run_one_day_clear_from_reader(
        config,
        dates,
        execution_dates,
        last_requested_date,
        date,
        read_quantity,
        read_next_columns,
        diagnostics_dir=diagnostics_dir,
    )

def run_backtest_clear_over_chunks(
    config: dict[str, Any], dates: list[str], diagnostics_dir: Path | None, jobs: int,
    day_runner: Callable[..., tuple[Frame, Frame, Frame, pd.Series]],
    params_builder: Callable[[list[str], list[str], str, str, Path | None], tuple[Any, ...]]) -> QuantityResult:
    """按日期 chunk 跑全区间回测，并汇总逐日结果。"""

    date_chunks = split_dates_for_jobs(config, dates, jobs)
    execution_dates = append_next_available_date(dates)
    last_requested_date = dates[-1]
    params = [
        params_builder(chunk, execution_dates, last_requested_date, date, diagnostics_dir)
        for chunk in date_chunks for date in chunk
    ]
    worker_count = worker_count_for_chunks(jobs, date_chunks)
    if worker_count > 1:
        with Pool(worker_count) as pool:
            day_results = pool.starmap(day_runner, params)
    if worker_count <= 1:
        day_results = [day_runner(*param) for param in params]
    pnl_list = [item[0] for item in day_results]
    turnover_list = [item[1] for item in day_results]
    cost_list = [item[2] for item in day_results]
    ep_holding_value_ratio_list = [item[3] for item in day_results]
    return build_result(config, dates, pnl_list, turnover_list, cost_list, ep_holding_value_ratio_list)

def run_one_day_clear_from_frames(
    config: dict[str, Any],
    dates: list[str],
    execution_dates: list[str],
    last_requested_date: str,
    date: str,
    ep_holding_by_date: dict[str, Frame],
    diagnostics_dir: Path | None = None,
) -> tuple[Frame, Frame, Frame, pd.Series]:
    """用内存 EP holding 跑一个交易日，其余计算口径与 DB 路径一致。"""

    def read_quantity(read_date: str) -> Frame:
        if read_date not in ep_holding_by_date:
            raise ValueError(f"missing in-memory ep_holding {read_date}")
        return ep_holding_by_date[read_date]

    def read_next_columns(next_date: str) -> pd.Index | None:
        if next_date not in ep_holding_by_date:
            return None
        return ep_holding_by_date[next_date].columns

    return run_one_day_clear_from_reader(config, dates, execution_dates, last_requested_date, date, read_quantity, read_next_columns, diagnostics_dir=diagnostics_dir)


def run_backtest_clear_from_frames(
    config: dict[str, Any], dates: list[str], ep_holding_by_date: dict[str, Frame], diagnostics_dir: Path | None = None, jobs: int = 1) -> QuantityResult:

    """用内存 EP holding 跑完整回测，核心计算与 `run_backtest_clear` 相同。"""

    def params_builder(
        chunk_dates: list[str],
        execution_dates: list[str],
        last_requested_date: str,
        date: str,
        diagnostics_dir: Path | None,
    ) -> tuple[Any, ...]:
        return (
            config,
            chunk_dates,
            execution_dates,
            last_requested_date,
            date,
            ep_holding_by_date,
            diagnostics_dir,
        )

    return run_backtest_clear_over_chunks(
        config=config,
        dates=dates,
        diagnostics_dir=diagnostics_dir,
        jobs=jobs,
        day_runner=run_one_day_clear_from_frames,
        params_builder=params_builder,
    )

def run_backtest_clear(config: dict[str, Any], dates: list[str], ep_holding_univ_name: str, ep_holding_key: str, 
    diagnostics_dir: Path | None = None, jobs: int = 1) -> QuantityResult:

    def params_builder(
        chunk_dates: list[str],
        execution_dates: list[str],
        last_requested_date: str,
        date: str,
        diagnostics_dir: Path | None,
    ) -> tuple[Any, ...]:
        return (config, chunk_dates, execution_dates, last_requested_date, date, ep_holding_univ_name, ep_holding_key, diagnostics_dir)

    return run_backtest_clear_over_chunks(
        config=config,
        dates=dates,
        diagnostics_dir=diagnostics_dir,
        jobs=jobs,
        day_runner=run_one_day_clear,
        params_builder=params_builder,
    )
