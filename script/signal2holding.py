import json5
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from sphinx.util.runtime_config import OKX_EXCHANGE, FREQ_1H, load_runtime_config, apply_runtime_config
from sphinx.util.exchange_api import read_universe, read_market_data, get_index, read_signal, get_dates, read_holding, write_holding, del_holding


FUNDING_BP_SCALE = 1e4
SUPPORTED_FUNDING_STAGE = "before_cs_norm_and_after_final_zero"


def parse_args() -> Any:
    parser = ArgumentParser(description="OKX 1h CS signal to pct holding.")
    parser.add_argument("config", type=Path, help="main2_cs_pct config path")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="reserved; only 1 is supported")
    parser.add_argument("--overwrite", action="store_true", help="delete output holding before writing")
    return parser.parse_args()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_non_empty_string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def require_float(mapping: dict[str, Any], key: str, label: str) -> float:
    value = mapping.get(key)
    assert value is not None
    if isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label}.{key} must be a finite number")
    return number


def configure_environment(config: dict[str, Any]) -> dict[str, Any]:
    if "runtime" in config:
        raise ValueError("config.runtime has been removed; set RUNTIME_CONFIG explicitly")

    runtime = load_runtime_config()
    # require_supported_runtime(runtime)
    apply_runtime_config(runtime, override=True)

    if config.get("expected_exchange") not in (None, runtime.get("EXCHANGE")):
        raise ValueError(f"runtime.EXCHANGE must be {config['expected_exchange']!r}, got {runtime.get('EXCHANGE')!r}")
    if config.get("expected_freq") not in (None, runtime.get("FREQ")):
        raise ValueError(f"runtime.FREQ must be {config['expected_freq']!r}, got {runtime.get('FREQ')!r}")
    if runtime.get("EXCHANGE") != OKX_EXCHANGE or runtime.get("FREQ") != FREQ_1H:
        raise ValueError("sig2holding only supports OKX 1h runtime")
    return runtime


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    require_mapping(config, "config")
    for key in ("start_date", "end_date", "nav", "holding_mode", "univ_name", "alpha_name", "out_name", "strategy"):
        if key not in config:
            raise ValueError(f"config missing required key: {key}")

    configure_environment(config)
    if config["holding_mode"] != "pct":
        raise ValueError(f"holding_mode must be 'pct', got {config['holding_mode']!r}")
    require_float(config, "nav", "config")
    for key in ("start_date", "end_date", "univ_name", "out_name"):
        require_non_empty_string(config, key, "config")

    alpha_name = config["alpha_name"]
    if not isinstance(alpha_name, list) or len(alpha_name) != 1:
        raise ValueError("alpha_name must be a single-item list")
    if not isinstance(alpha_name[0], str) or not alpha_name[0]:
        raise ValueError("alpha_name[0] must be a non-empty string")

    params = require_mapping(require_mapping(config["strategy"], "strategy").get("params"), "strategy.params")
    if params.get("transform", "cs_demean_l1") != "cs_demean_l1":
        raise ValueError(f"strategy.params.transform must be 'cs_demean_l1', got {params.get('transform')!r}")

    smoothing = require_mapping(params.get("smoothing"), "strategy.params.smoothing")
    if smoothing.get("type") != "ewma":
        raise ValueError(f"strategy.params.smoothing.type must be 'ewma', got {smoothing.get('type')!r}")
    alpha = require_float(smoothing, "alpha", "strategy.params.smoothing")
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("strategy.params.smoothing.alpha must be in (0, 1]")

    funding = require_mapping(params.get("funding_filter"), "strategy.params.funding_filter")
    if funding.get("stage") != SUPPORTED_FUNDING_STAGE:
        raise ValueError("strategy.params.funding_filter.stage must be {SUPPORTED_FUNDING_STAGE!r}, got {funding.get('stage')!r}")
    require_non_empty_string(funding, "feature", "strategy.params.funding_filter")
    if require_float(funding, "threshold_bp", "strategy.params.funding_filter") < 0.0:
        raise ValueError("strategy.params.funding_filter.threshold_bp must be non-negative")
    return config


def read_feature_frame(date: str, insts: pd.Index, feature_name: str) -> pd.DataFrame:
    return pd.concat([read_market_data(date, inst, f"feature/{feature_name}").rename(inst) for inst in insts], axis=1)


def read_signal_and_funding(
    date: str,
    univ_name: str,
    signal_key: str,
    funding_feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = read_universe(date, univ_name)
    signal = read_signal(date, univ_name, signal_key).reindex(
        index=get_index(date),
        columns=universe.index,
    )
    if signal.isna().any().any():
        raise ValueError(f"signal has {int(signal.isna().sum().sum())} NaNs after universe alignment on {date}")
    funding = read_feature_frame(date, universe.index, funding_feature).reindex(
        index=signal.index,
        columns=signal.columns,
    )
    return signal.astype(float), funding.fillna(0.0).astype(float)


def normalize_after_funding_prefilter(
    signal: pd.DataFrame,
    funding: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    funding = funding.reindex(index=signal.index, columns=signal.columns).fillna(0.0).astype(float)
    initial_demeaned = signal.sub(signal.mean(axis=1), axis=0)
    blocked = initial_demeaned.mul(funding).gt(0.0) & funding.abs().ge(threshold)

    filtered = signal.mask(blocked)
    demeaned = filtered.sub(filtered.mean(axis=1), axis=0)
    denom = demeaned.abs().sum(axis=1)
    normalized = demeaned.div(denom.replace(0.0, np.nan), axis=0).fillna(0.0)
    return normalized.mask(blocked, 0.0).fillna(0.0)


def zero_adverse_funding_positions(holding: pd.DataFrame, funding: pd.DataFrame, threshold: float) -> pd.DataFrame:
    funding = funding.reindex(index=holding.index, columns=holding.columns).fillna(0.0).astype(float)
    blocked = holding.mul(funding).gt(0.0) & funding.abs().ge(threshold)
    return holding.mask(blocked, 0.0).fillna(0.0)


def make_pct_holdings(
    dates: list[str],
    signals: dict[str, pd.DataFrame],
    fundings: dict[str, pd.DataFrame],
    alpha: float,
    funding_threshold: float,
) -> dict[str, pd.DataFrame]:
    normalized = [
        normalize_after_funding_prefilter(signals[date], fundings[date], funding_threshold)
        for date in dates
    ]
    smoothed = pd.concat(normalized).fillna(0.0).ewm(alpha=alpha, adjust=False).mean()

    holdings = {}
    for date in dates:
        signal = signals[date]
        holding = smoothed.reindex(index=signal.index, columns=signal.columns).fillna(0.0)
        holdings[date] = zero_adverse_funding_positions(holding, fundings[date], funding_threshold)
    return holdings


def generate_holdings(config: dict[str, Any], dates: list[str]) -> dict[str, pd.DataFrame]:
    params = config["strategy"]["params"]
    funding_filter = params["funding_filter"]
    univ_name = config["univ_name"]
    signal_key = config["alpha_name"][0]
    funding_feature = funding_filter["feature"]
    threshold = float(funding_filter["threshold_bp"]) / FUNDING_BP_SCALE

    signals = {}
    fundings = {}
    for date in tqdm(dates, desc="load"):
        signals[date], fundings[date] = read_signal_and_funding(date, univ_name, signal_key, funding_feature)

    holdings = make_pct_holdings(dates, signals, fundings, alpha=float(params["smoothing"]["alpha"]), funding_threshold=threshold)
    for date in tqdm(dates, desc="write"):
        write_holding(date, univ_name, config["out_name"], holdings[date])
    return holdings


def require_output_holding_absent(dates: list[str], univ_name: str, holding_key: str) -> None:
    for date in dates:
        try:
            read_holding(date, univ_name, holding_key)
        except KeyError:
            continue
        raise ValueError(f"output holding key already exists at {date}: {univ_name}/{holding_key}")


def maybe_delete_output_holding(univ_name: str, holding_key: str, overwrite: bool) -> None:
    if not overwrite:
        return
    try:
        del_holding(univ_name, holding_key)
    except (FileNotFoundError, KeyError):
        pass


def main() -> None:
    args = parse_args()
    if args.jobs != 1:
        raise ValueError("main2_cs_pct is stateful EWMA code; only -j 1 is supported")

    with args.config.open("r", encoding="utf-8") as f:
        config = validate_config(json5.load(f))

    dates = get_dates(config["start_date"], config["end_date"])
    if not dates:
        raise SystemExit("no selected dates")

    univ_name = config["univ_name"]
    out_name = config["out_name"]

    if args.overwrite:
        maybe_delete_output_holding(univ_name, out_name, overwrite=True)
    else:
        require_output_holding_absent(dates, univ_name, out_name)

    holdings = generate_holdings(config, dates)
    print(f"holding_univ_name {univ_name}")
    print(f"holding_key {out_name}")
    print(f"dates {dates[0]} {dates[-1]} {len(dates)}")


if __name__ == "__main__":
    main()
