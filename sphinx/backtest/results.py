import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import datetime as dt


from .config import annual_periods_for_exchange


Frame = pd.DataFrame
EPS = 1e-5


@dataclass
class QuantityResult:
    """回测汇总结果, 这个类只放最终输出会用到的几个对象。逐 bar 诊断另外写 parquet """
    metrics: pd.DataFrame
    daily_pnl: pd.Series
    daily_turnover: pd.Series
    minute_equity: pd.Series
    all_cost: float
    product_daily_pnl: Frame | None = None


def safe_div(numerator: float, denominator: float) -> float:
    if pd.isna(denominator) or abs(denominator) < EPS:
        return np.nan
    return numerator / denominator


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    """标准回撤：当前净值减去到当前为止的历史最高净值。

    初始净值 1.0 必须放进去，否则第一天或第一根柱子先亏时，
    历史最高净值会被错误地当成亏损后的净值。
    """

    equity_with_start = pd.concat(
        [pd.Series([1.0]), equity.reset_index(drop=True)],
        ignore_index=True,
    )
    return equity_with_start - equity_with_start.cummax()


def scaled_instrument_daily_pnl(dates: list[str], pnl_list: list[Frame]) -> pd.Series:
    """按日期和品种展开，并把单品种贡献还原到当天 universe 尺度。"""

    series_list: list[pd.Series] = []
    for date, pnl in zip(dates, pnl_list, strict=True):
        daily_inst_pnl = pnl.sum(axis=0) * len(pnl.columns)
        daily_inst_pnl.index = pd.MultiIndex.from_product([[pd.Timestamp(date)], daily_inst_pnl.index], names=["date", "instrument"])
        series_list.append(daily_inst_pnl)
    if not series_list:
        return pd.Series(dtype=float)
    return pd.concat(series_list)


def annualized_sharpe(series: pd.Series, periods: int) -> float:
    std = series.std()
    if pd.isna(std) or abs(std) < EPS:
        return np.nan
    return series.mean() / std * np.sqrt(periods)


def turnover_implied_avg_holding_bars(
    ep_holding_value_ratio: pd.Series | None,
    total_turnover: float,
) -> float:
    """资金加权平均持仓 bar 数:2 * 持仓市值存量 / 成交流量。

    回测里的 turnover 是双边成交额 / nav。一个完整持仓 episode 有开仓和
    平仓两边成交，所以用 2 倍持仓市值 bar 存量除以总成交额。
    """

    if ep_holding_value_ratio is None or ep_holding_value_ratio.empty:
        return np.nan
    return safe_div(2.0 * float(ep_holding_value_ratio.sum()), total_turnover)


def build_result(
    config: dict[str, Any], dates: list[str], pnl_list: list[Frame], turnover_list: list[Frame], cost_list: list[Frame],
    ep_holding_value_ratio_list: list[pd.Series] | None = None,
) -> QuantityResult:
    """把逐日结果汇总成指标表。"""

    minute_pnl = pd.concat([pnl.sum(axis=1) for pnl in pnl_list])
    minute_equity = minute_pnl.cumsum() + 1
    daily_index = pd.to_datetime(dates)
    daily_pnl = pd.Series([pnl.sum().sum() for pnl in pnl_list], index=daily_index)
    product_daily_pnl = pd.DataFrame(
        [pnl.sum(axis=0) for pnl in pnl_list],
        index=daily_index,
    ).fillna(0.0)
    inst_daily_pnl = scaled_instrument_daily_pnl(dates, pnl_list)
    daily_turnover = pd.Series([turnover.sum().sum() for turnover in turnover_list], index=daily_index)

    weekly_returns = daily_pnl.resample("W").sum()
    monthly_returns = daily_pnl.resample("ME").sum()
    equity = daily_pnl.cumsum() + 1
    max_drawdown = drawdown_from_equity(equity).min()
    minute_max_drawdown = drawdown_from_equity(minute_equity).min()

    total_pnl = daily_pnl.sum()
    total_turnover = daily_turnover.sum()
    all_cost = sum(cost.sum().sum() for cost in cost_list)
    ep_holding_value_ratio = None
    max_ep_holding_value_ratio = np.nan
    if ep_holding_value_ratio_list:
        ep_holding_value_ratio = pd.concat(ep_holding_value_ratio_list)
        max_ep_holding_value_ratio = ep_holding_value_ratio.max()
    avg_holding_bars = turnover_implied_avg_holding_bars(ep_holding_value_ratio, total_turnover)
    annual_periods = annual_periods_for_exchange(os.environ.get("EXCHANGE", ""))
    annualized_pnl = total_pnl / len(daily_pnl) * annual_periods
    metrics = pd.DataFrame(
        {
            "sharp": [annualized_sharpe(daily_pnl, annual_periods)],
            "inst_daily_sharp": [annualized_sharpe(inst_daily_pnl, annual_periods)],
            "weekly_sharp": [annualized_sharpe(weekly_returns, 52)],
            "monthly_sharp": [annualized_sharpe(monthly_returns, 12)],
            "pnl": [total_pnl],
            "calmar": [safe_div(annualized_pnl, -max_drawdown)],
            "turnover": [daily_turnover.mean()],
            "max_drawdown": [max_drawdown],
            "minute_max_drawdown": [minute_max_drawdown],
            "max_ep_holding_value_ratio": [max_ep_holding_value_ratio],
            "avg_holding_bars": [avg_holding_bars],
            "avg_holding_bars_by_turnover": [avg_holding_bars],
            "margin": [safe_div(total_pnl * 1e4, total_turnover)],
            "cost": [safe_div(all_cost * 1e4, total_turnover)],
            "win_rate": [(daily_pnl > 1e-4).mean()],
            "weekly_win_rate": [(weekly_returns > 1e-4).mean()],
            "monthly_win_rate": [(monthly_returns > 1e-4).mean()],
        },
        index=[f"nav {int(config['nav'] / 1e4)}w USDT"],
    )
    return QuantityResult(metrics, daily_pnl, daily_turnover, minute_equity, all_cost, product_daily_pnl)


def write_outputs(result: QuantityResult, output_dir: Path) -> None:
    """写出当前回测结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(output_dir / "metrics.csv")
    result.daily_pnl.rename("pnl").to_csv(output_dir / "daily_pnl.csv")
    result.daily_turnover.rename("turnover").to_csv(output_dir / "daily_turnover.csv")
    result.minute_equity.rename("equity").to_csv(output_dir / "minute_equity.csv")
    if result.product_daily_pnl is not None:
        result.product_daily_pnl.to_csv(output_dir / "product_daily_pnl.csv")


def select_product_daily_pnl_for_plot(product_daily_pnl: Frame, keep: int = 5) -> Frame:
    """Plot only the best and worst products by total PnL."""

    if product_daily_pnl.empty:
        return product_daily_pnl
    final_pnl = product_daily_pnl.fillna(0.0).sum(axis=0)
    best = final_pnl.sort_values(ascending=False).head(keep).index.tolist()
    worst = final_pnl.sort_values(ascending=True).head(keep).index.tolist()
    columns = list(dict.fromkeys(best + worst))
    return product_daily_pnl.loc[:, columns]


def plot_x_limits(result: QuantityResult, product_daily: Frame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if len(result.minute_equity.index) > 0:
        left = pd.Timestamp(result.minute_equity.index.min())
        right = pd.Timestamp(result.minute_equity.index.max())
    elif len(result.daily_turnover.index) > 0:
        left = pd.Timestamp(result.daily_turnover.index.min())
        right = pd.Timestamp(result.daily_turnover.index.max()) + pd.Timedelta(days=1)
    elif len(product_daily.index) > 0:
        left = pd.Timestamp(product_daily.index.min())
        right = pd.Timestamp(product_daily.index.max()) + pd.Timedelta(days=1)
    else:
        return None
    if left < right:
        return left, right
    elif left == right:
        return left, right + pd.Timedelta(days=1)
    else:
        raise ValueError(f"invalid plot x limits: {left} > {right}")


def plot_header_text(path: Path, created_at: str) -> str:
    return f"image_path: {path.resolve()}\ncreated_at: {created_at}"


def write_plot(result: QuantityResult, output_dir: Path, plot_path: Path | None = None) -> Path:
    """可选输出回测曲线：自身 equity、每日换手、分品种累计 PnL。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = plot_path or output_dir / "result.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    result.minute_equity.plot(ax=axes[0], title="minute equity", linewidth=1.0)
    axes[1].bar(result.daily_turnover.index, result.daily_turnover.to_numpy(), width=1.0, align="edge")
    axes[1].set_title("daily turnover")

    product_daily = pd.DataFrame()
    if result.product_daily_pnl is not None:
        product_daily = select_product_daily_pnl_for_plot(result.product_daily_pnl)
    axes[2].set_title("product cumulative pnl unavailable")
    product_handles: list[Any] = []
    product_labels: list[str] = []
    if not product_daily.empty:
        product_daily.cumsum().plot(ax=axes[2], linewidth=0.9, legend=False)
        axes[2].set_title("product cumulative pnl top/bottom 5")
        product_handles, product_labels = axes[2].get_legend_handles_labels()

    for ax in axes:
        ax.grid(True, alpha=0.25)
    x_limits = plot_x_limits(result, product_daily)
    if x_limits is None:
        for ax in axes:
            ax.margins(x=0)
    else:
        for ax in axes:
            ax.margins(x=0)
            ax.set_xlim(x_limits)
    created_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.9, bottom=0.15, hspace=0.36)
    fig.text(0.01, 0.985, plot_header_text(path, created_at), ha="left", va="top", fontsize=7)
    if product_handles:
        fig.legend(
            product_handles,
            product_labels,
            ncol=min(5, len(product_labels)),
            fontsize=6,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            frameon=False,
        )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
