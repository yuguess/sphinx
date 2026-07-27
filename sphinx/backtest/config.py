import json5
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sphinx.util.runtime_config import load_runtime_config, apply_runtime_config, require_supported_runtime, OKX_EXCHANGE, CF_EXCHANGE
from sphinx.backtest.validation import require_positive_number, require_config_value, require_nonempty_string, require_mapping
from sphinx.backtest.validation import require_nonnegative_number, require_bool


SUPPORTED_EP_HOLDING_MODES = {"qty", "pct"}


PARALLEL_CHUNK_DAYS_BY_EXCHANGE = {
    CF_EXCHANGE: 65,
    OKX_EXCHANGE: 65,
}


CAPACITY_VIOLATION_TOLERANCE = 1e-5


@dataclass(frozen=True)
class BacktestSettings:
    """回测输入和输出设置。所有字段都必须来自 config。"""
    ep_holding_univ_name: str
    ep_holding_key: str
    ep_holding_mode: str
    output_dir: Path
    diagnostics_dir: Path | None
    compare_dir: Path | None
    fee_cache: Path | None
    plot: bool
    plot_path: Path | None


def require_parallel_chunk_supported(config: dict[str, Any]) -> int:
    runtime = runtime_from_config(config)
    require_supported_runtime(runtime, error_type=SystemExit)
    return PARALLEL_CHUNK_DAYS_BY_EXCHANGE[str(runtime["EXCHANGE"])]


def split_dates_by_chunk_len(dates: list[str], chunk_len: int) -> list[list[str]]:
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    return [dates[i:i + chunk_len] for i in range(0, len(dates), chunk_len)]


def split_dates_for_jobs(config: dict[str, Any], dates: list[str], jobs: int) -> list[list[str]]:
    if jobs == -1:
        return split_dates_by_chunk_len(dates, require_parallel_chunk_supported(config))
    if jobs <= 1:
        return [dates]
    require_parallel_chunk_supported(config)
    chunk_len = (len(dates) - 1) // jobs + 1
    return split_dates_by_chunk_len(dates, chunk_len)


def runtime_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """读取外部 runtime 配置；业务 config 不允许内嵌 runtime。"""
    if "runtime" in config:
        raise SystemExit("config.runtime 已删除，请使用 RUNTIME_CONFIG 指定外部 runtime")
    try:
        return load_runtime_config()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def assert_environment_matches_config(config: dict[str, Any]) -> None:
    """入口之后环境必须完全等于外部 runtime。"""

    runtime = runtime_from_config(config)
    require_supported_runtime(runtime, error_type=SystemExit)
    for key in ["EXCHANGE", "FREQ", "MARKET", "DATA_ROOT", "DEPLOY_MODE", "USE_MP_CACHE", "ENV_FILE"]:
        if os.environ[key] != str(runtime[key]):
            raise SystemExit(f"runtime.{key} was changed after config load")


def configure_environment(config: dict[str, Any]) -> None:
    """用 RUNTIME_CONFIG 指向的外部 runtime 设置运行环境。"""
    try:
        apply_runtime_config(runtime_from_config(config), override=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    assert_environment_matches_config(config)


def required_backtest_string(backtest: dict[str, Any], key: str) -> str:
    """读取必填 backtest 字符串字段。"""

    if key not in backtest:
        raise SystemExit(f"backtest missing required key: {key}")
    return require_nonempty_string(backtest[key], f"backtest.{key}")


def strategy_exec_info(config: dict[str, Any]) -> dict[str, Any]:
    """取出策略执行配置，同时做深层结构检查。"""

    strategy = require_mapping(config.get("strategy"), "strategy")
    params = require_mapping(strategy.get("params"), "strategy.params")
    return require_mapping(params.get("exec_info"), "strategy.params.exec_info")


def validate_supported_config(config: dict[str, Any]) -> None:
    """只允许当前审过的 native quantity + 5min + make 路径通过。"""

    require_positive_number(config.get("nav"), "nav")

    runtime = runtime_from_config(config)
    require_supported_runtime(runtime, error_type=SystemExit)
    if "expected_exchange" in config:
        require_config_value(runtime, "EXCHANGE", require_nonempty_string(config["expected_exchange"], "expected_exchange"), "runtime")
    for key in ["DATA_ROOT", "DEPLOY_MODE", "USE_MP_CACHE", "ENV_FILE"]:
        if key not in runtime:
            raise SystemExit(f"runtime missing required key: {key}")

    backtest = require_mapping(config.get("backtest"), "backtest")
    if "quantity_dir" in backtest:
        raise SystemExit("backtest.quantity_dir 已删除，请使用 ep_holding_univ_name 和 ep_holding_key")
    deprecated_keys = {
        "holding_univ_name": "ep_holding_univ_name",
        "holding_key": "ep_holding_key",
    }
    for old_key, new_key in deprecated_keys.items():
        if old_key in backtest:
            if new_key in backtest and backtest[old_key] != backtest[new_key]:
                raise SystemExit(f"backtest.{old_key} conflicts with backtest.{new_key}")
            backtest[new_key] = backtest[old_key]
    for key in ["output_dir", "ep_holding_univ_name", "ep_holding_key"]:
        required_backtest_string(backtest, key)

    ep_holding_mode = ep_holding_mode_from_config(config)

    if "fee_cache" in backtest:
        require_nonempty_string(backtest["fee_cache"], "backtest.fee_cache")
    exchange = runtime.get("EXCHANGE")

    if exchange == OKX_EXCHANGE:
        require_nonnegative_number(config.get("fee"), "fee")
    elif exchange == CF_EXCHANGE:
        pass
    else:
        raise SystemExit(f"only {CF_EXCHANGE}/{OKX_EXCHANGE} runtime is supported, got {exchange!r}")

    if ep_holding_mode == "pct" and exchange != OKX_EXCHANGE:
        raise SystemExit("backtest.ep_holding_mode='pct' only supports runtime.EXCHANGE='okx'")
    if "plot" in backtest:
        require_bool(backtest["plot"], "backtest.plot")
    if "plot_path" in backtest:
        require_nonempty_string(backtest["plot_path"], "backtest.plot_path")
    if "repair_invalid_bar_holding" in backtest:
        require_bool(backtest["repair_invalid_bar_holding"], "backtest.repair_invalid_bar_holding")
    if "capacity_violation_tolerance" in backtest:
        require_nonnegative_number(backtest["capacity_violation_tolerance"], "backtest.capacity_violation_tolerance")

    exec_info = strategy_exec_info(config)
    exec_type = exec_info.get("exec_type")
    if exec_type != "make":
        raise SystemExit(f"strategy.params.exec_info.exec_type must be 'make', got {exec_type!r}")
    if "extra_slippage" in exec_info:
        raise SystemExit("strategy.params.exec_info.extra_slippage 已删除，不再支持")
    if "turnover_limit_rate" not in exec_info:
        raise SystemExit("strategy.params.exec_info missing required key: turnover_limit_rate")
    require_positive_number(exec_info["turnover_limit_rate"], "strategy.params.exec_info.turnover_limit_rate")


def ep_holding_mode_from_config(config: dict[str, Any]) -> str:
    backtest = require_mapping(config.get("backtest"), "backtest")
    mode = backtest.get("ep_holding_mode", "qty")
    if mode not in SUPPORTED_EP_HOLDING_MODES:
        raise SystemExit(f"backtest.ep_holding_mode must be one of {sorted(SUPPORTED_EP_HOLDING_MODES)}, got {mode!r}")
    return str(mode)


def backtest_settings(config: dict[str, Any]) -> BacktestSettings:
    backtest = config["backtest"]
    diagnostics_dir = backtest.get("diagnostics_dir")
    compare_dir = backtest.get("compare_dir")
    fee_cache = backtest.get("fee_cache")
    plot_path = backtest.get("plot_path")
    diagnostics_path = None
    compare_path = None
    fee_cache_path = None
    resolved_plot_path = None
    if diagnostics_dir:
        diagnostics_path = Path(diagnostics_dir)
    if compare_dir:
        compare_path = Path(compare_dir)
    if fee_cache:
        fee_cache_path = Path(fee_cache)
    if plot_path:
        resolved_plot_path = Path(plot_path)
    return BacktestSettings(
        ep_holding_univ_name=required_backtest_string(backtest, "ep_holding_univ_name"),
        ep_holding_key=required_backtest_string(backtest, "ep_holding_key"),
        ep_holding_mode=ep_holding_mode_from_config(config),
        output_dir=Path(backtest["output_dir"]),
        diagnostics_dir=diagnostics_path,
        compare_dir=compare_path,
        fee_cache=fee_cache_path,
        plot=bool(backtest.get("plot", False)),
        plot_path=resolved_plot_path,
    )


def load_config(path: Path) -> dict[str, Any]:
    """读取 json5 配置，并检查回测需要的基本字段。"""

    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    with path.open("r") as f:
        config = json5.load(f)
    if not isinstance(config, dict):
        raise SystemExit("config root must be an object")
    for key in ["start_date", "end_date", "nav", "strategy", "backtest"]:
        if key not in config:
            raise SystemExit(f"config missing required key: {key}")
    validate_supported_config(config)
    return config


def annual_periods_for_exchange(exchange: str) -> int:
    if exchange == OKX_EXCHANGE:
        return 365
    elif exchange == CF_EXCHANGE:
        return 240
    else:
        raise SystemExit(f"only {CF_EXCHANGE}/{OKX_EXCHANGE} annual periods are supported, got {exchange!r}")


def worker_count_for_chunks(jobs: int, date_chunks: list[list[str]]) -> int:
    if not date_chunks:
        return 1
    if jobs == -1:
        return len(date_chunks)
    return min(max(1, jobs), len(date_chunks))


def next_selected_date(dates: list[str], date: str) -> str | None:
    """返回交易日列表里的下一个交易日；列表末尾返回 None。"""

    index = dates.index(date)
    if index < len(dates) - 1:
        return dates[index + 1]
    elif index == len(dates) - 1:
        return None
    else:
        raise ValueError(f"{date} index is outside selected dates")

def require_make_execution(config: dict[str, Any]) -> dict[str, Any]:
    """这个清晰版只实现 make。"""
    exec_info = strategy_exec_info(config)
    exec_type = exec_info.get("exec_type")
    if exec_type != "make":
        raise SystemExit("strategy.params.exec_info.exec_type must be 'make'")
    if "extra_slippage" in exec_info:
        raise SystemExit("strategy.params.exec_info.extra_slippage 已删除，不再支持")
    return exec_info


def should_repair_invalid_bar_holding(config: dict[str, Any]) -> bool:
    return bool(config.get("backtest", {}).get("repair_invalid_bar_holding", False))


def previous_selected_date(dates: list[str], date: str) -> str:
    """返回已选择交易日里的前一个交易日。"""

    index = dates.index(date)
    if index <= 0:
        raise ValueError(f"{date} has no previous selected date")
    return dates[index - 1]


def configured_capacity_violation_tolerance(config: dict[str, Any]) -> float:
    backtest = config.get("backtest", {})
    if "capacity_violation_tolerance" in backtest:
        return require_nonnegative_number(
            backtest["capacity_violation_tolerance"],
            "backtest.capacity_violation_tolerance",
        )
    return CAPACITY_VIOLATION_TOLERANCE
