import time
import os
import math
import argparse
import json
import pandas as pd
import numpy as np
import copy
import torch
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sphinx.core.model import gen_model
from mona.common.util import dbtool
from mona.common import SHM_ROOT, shm, metadata
from mona.sdk import SHMContext


DEFAULT_SDK_FREQ = "1h"
DEFAULT_MODEL_DIR = Path(
    "output_deploy/okx1h/"
    "universe_t60r30_oi_list30d_bnokxwithspot_okx_futures_1h_return24_global_label_corr_20241231_flip0_lr1e5_step1_gamma0p8_bsz8"
)
DEFAULT_UNIVERSE = "universe_t60r30_oi_list30d_bnokxwithspot_okx_futures"
DEFAULT_SIGNAL_KEY = "24h_global_label_corr_20241231_flip0_lr1e5_gamma0p8_bsz8_epoch9fusion_hk2_shadow"
DEFAULT_AUDIT_DIR = Path("deploy/data/okx1h_signal_only_audit")
SPECULATIVE_PREWARM_TIMEOUT_SECONDS = 0.0


class UnsafeRealWriteError(RuntimeError):
    pass


class LiveDataUnavailable(RuntimeError):
    pass


class ShmAccessUnavailable(LiveDataUnavailable):
    pass


@dataclass(frozen=True)
class FeaturePlan:
    alphas: list[str]
    seq_len: int
    in_channels: int
    out_channels: int
    label_name: Any
    model_name: str


@dataclass(frozen=True)
class FeatureWindow:
    values: np.ndarray
    columns: list[str]
    index: pd.DatetimeIndex
    selected_time: pd.Timestamp
    requested_time: pd.Timestamp | None
    task_time_floor_used: bool
    feature_lengths: dict[str, int]
    source_date: str | None = None


@dataclass(frozen=True)
class InferenceResult:
    signal: pd.DataFrame
    selected_time: pd.Timestamp
    feature_window: FeatureWindow
    version_stats: list[dict[str, Any]]


@dataclass(frozen=True)
class CheckpointInfo:
    version: str
    path: str
    load_ok: bool
    top_level_keys: list[str]
    config: dict[str, Any]
    feature_plan: FeaturePlan
    state_dict_key: str
    state_dict_summary: dict[str, Any]
    normalizer_summary: dict[str, Any]
    output_semantics: dict[str, Any]


class LoopState:
    def __init__(self, last_written_selected_time: pd.Timestamp | None = None) -> None:
        self.last_written_selected_time = last_written_selected_time
        self.last_feature_window: FeatureWindow | None = None

    def should_write(self, selected_time: pd.Timestamp) -> bool:
        return self.last_written_selected_time != selected_time

    def mark_written(self, selected_time: pd.Timestamp) -> None:
        self.last_written_selected_time = selected_time


def _as_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _label_to_str(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _extract_frame(frame: Any) -> tuple[list[str], list[str], list[list[Any]]]:
    if isinstance(frame, dict):
        columns = [_label_to_str(c) for c in frame["columns"]]
        index = [_label_to_str(i) for i in frame["index"]]
        data = [list(row) for row in frame["data"]]
    else:
        data_values = getattr(frame, "values", frame)
        data = _as_list(data_values)
        data = [list(row) for row in data]

        raw_columns = getattr(frame, "columns", range(len(data[0]) if data else 0))
        raw_index = getattr(frame, "index", range(len(data)))
        columns = [_label_to_str(c) for c in _as_list(raw_columns)]
        index = [_label_to_str(i) for i in _as_list(raw_index)]

    if len(index) != len(data):
        raise ValueError(f"index length mismatch: {len(index)} vs {len(data)} rows")
    if any(len(row) != len(columns) for row in data):
        raise ValueError("row width mismatch against columns")
    return columns, index, data


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return None
    return number


def _numeric_stats(values: list[Any]) -> dict[str, Any]:
    finite_values = []
    nan_count = 0
    inf_count = 0
    non_numeric_count = 0

    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            non_numeric_count += 1
            continue
        if math.isnan(number):
            nan_count += 1
            continue
        if math.isinf(number):
            inf_count += 1
            continue
        finite_values.append(number)

    stats: dict[str, Any] = {
        "count": len(values),
        "finite_count": len(finite_values),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "non_numeric_count": non_numeric_count,
    }
    if finite_values:
        mean = sum(finite_values) / len(finite_values)
        variance = sum((x - mean) ** 2 for x in finite_values) / len(finite_values)
        stats.update(
            {
                "min": min(finite_values),
                "max": max(finite_values),
                "mean": mean,
                "std": variance**0.5,
                "sum": sum(finite_values),
                "abs_sum": sum(abs(x) for x in finite_values),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "sum": None, "abs_sum": None})
    return stats


def summarize_signal_frame(frame: Any, sample_rows: int = 5, sample_cols: int = 8) -> dict[str, Any]:
    columns, index, data = _extract_frame(frame)
    flat_values = [value for row in data for value in row]

    column_stats = {}
    for col_idx, col in enumerate(columns):
        column_stats[col] = _numeric_stats([row[col_idx] for row in data])

    sampled_columns = columns[:sample_cols]
    row_sample = []
    for row_idx, row in enumerate(data[:sample_rows]):
        sample = {"index": index[row_idx]}
        for col_idx, col in enumerate(sampled_columns):
            sample[col] = _json_value(row[col_idx])
        row_sample.append(sample)

    return {
        "shape": [len(index), len(columns)],
        "index_start": index[0] if index else None,
        "index_end": index[-1] if index else None,
        "columns": columns,
        "columns_count": len(columns),
        "stats": _numeric_stats(flat_values),
        "stats_by_column": column_stats,
        "sample": {
            "rows": min(sample_rows, len(index)),
            "columns": min(sample_cols, len(columns)),
            "data": row_sample,
        },
    }


class AuditSignalWriter:
    """Local audit writer only; it never calls DB, holding, portfolio, or execution APIs."""

    def __init__(self, output_dir: str | Path = DEFAULT_AUDIT_DIR) -> None:
        self.output_dir = Path(output_dir)

    def write(
        self, *, date: str, universe_name: str, signal_key: str, signal: pd.DataFrame, versions: list[str],
        feature_plan: FeaturePlan, task_time: str, extra: dict[str, Any] | None = None,
    ) -> Path:
        audit = summarize_signal_frame(signal)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out_dir = self.output_dir / signal_key / date / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            "audit_version": 2,
            "mode": "signal_only_dry_run",
            "real_db_write": False,
            "holding_write": False,
            "portfolio_optimization": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "date": date,
            "task_time": task_time,
            "universe": universe_name,
            "signal_key": signal_key,
            "versions": versions,
            "version_count": len(versions),
            "feature_info": _jsonable(feature_plan.__dict__),
            "feature_count": len(feature_plan.alphas),
            **audit,
            "extra": extra or {},
        }
        if extra and "selected_time" in extra:
            payload["selected_time"] = extra["selected_time"]
        path = out_dir / "audit.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        signal.to_csv(out_dir / "signal.csv")
        return path


class SDKSignalWriter:
    """Write a single-row signal frame to quant-dev f2.sdk SHM signal."""

    def __init__(self, context_factory: Any | None = None) -> None:
        self.context_factory = context_factory or self._default_context_factory
        self.contexts: dict[str, Any] = {}

    @staticmethod
    def _default_context_factory(date: str) -> Any:
        return SHMContext(date, wait_init=False)

    def _close_context(self, date: str) -> None:
        ctx = self.contexts.pop(date, None)
        if ctx is None:
            return
        close = getattr(ctx, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            print(f"SDK_CONTEXT_CLOSE_FAILED date={date}: {exc}", flush=True)

    def _close_stale_contexts(self, current_date: str) -> None:
        for date in list(self.contexts):
            if date != current_date:
                self._close_context(date)

    def _context(self, date: str) -> Any:
        self._close_stale_contexts(date)
        if date not in self.contexts:
            try:
                self.contexts[date] = self.context_factory(date)
            except Exception as exc:
                raise LiveDataUnavailable(f"cannot open signal SHM context for {date}: {exc}") from exc
        return self.contexts[date]

    def close(self) -> None:
        for date in list(self.contexts):
            self._close_context(date)

    @staticmethod
    def _timestamp_ns(value: Any) -> int:
        timestamp = pd.Timestamp(value)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize("Asia/Shanghai")
        else:
            timestamp = timestamp.tz_convert("Asia/Shanghai")
        return int(timestamp.value)

    @staticmethod
    def _signal_index(ctx: Any, timestamp: pd.Timestamp) -> int:
        target = SDKSignalWriter._timestamp_ns(timestamp)
        index_ts = np.asarray(ctx.index_ts, dtype=np.int64)
        positions = np.flatnonzero(index_ts == target)
        if len(positions) == 0:
            raise LiveDataUnavailable(f"signal timestamp {timestamp} is not available in SHM index")
        if len(positions) > 1:
            raise ValueError(f"signal timestamp {timestamp} does not map to exactly one SHM index")
        return int(positions[0])

    @staticmethod
    def _universe(ctx: Any) -> pd.Index:
        read_universe = getattr(ctx, "read_universe", None)
        if callable(read_universe):
            return pd.Index([str(symbol) for symbol in read_universe()])
        return pd.Index([str(symbol) for symbol in ctx.univ_symbols])

    @staticmethod
    def _series_for_context(signal: pd.DataFrame, universe: pd.Index) -> pd.Series:
        if len(signal.index) != 1:
            raise ValueError(f"SDK signal writer expects one signal row, got {len(signal.index)}")
        columns = pd.Index([str(column) for column in signal.columns])
        if columns.empty:
            raise ValueError("signal must contain at least one column")
        if not columns.is_unique:
            raise ValueError("signal columns must be unique")
        missing = [column for column in columns if column not in set(universe)]
        if missing:
            raise ValueError(f"signal columns are not in SHM universe: {missing[:8]}")
        series = signal.iloc[0].copy()
        series.index = columns
        series = series.astype(float)
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("signal contains nan or inf")
        expanded = pd.Series(np.nan, index=universe, dtype=float)
        expanded.loc[columns] = series.reindex(columns).to_numpy(dtype=float)
        return expanded

    @staticmethod
    def _signal_versions(ctx: Any) -> Any | None:
        sh = getattr(ctx, "sh", None)
        version = getattr(sh, "version", None)
        if not callable(version):
            return None
        return version("signal")

    @staticmethod
    def _require_complete_signal_prefix(ctx: Any, i: int) -> None:
        versions = SDKSignalWriter._signal_versions(ctx)
        if versions is None:
            return
        missing = np.flatnonzero(np.asarray(versions[:i]) == -1).tolist()
        if missing:
            raise LiveDataUnavailable(f"cannot write signal slot {i}; missing previous signal slots: {missing}")

    def write(
        self, *, date: str, universe_name: str, signal_key: str, signal: pd.DataFrame, versions: list[str], feature_plan: FeaturePlan,
        task_time: str, extra: dict[str, Any] | None = None) -> str:

        context_date = str((extra or {}).get("source_date") or date)
        ctx = self._context(context_date)
        try:
            i = self._signal_index(ctx, pd.Timestamp(signal.index[0]))
            universe = self._universe(ctx)
        except LiveDataUnavailable:
            self._close_context(context_date)
            raise
        except ValueError:
            self._close_context(context_date)
            raise
        except Exception as exc:
            self._close_context(context_date)
            raise LiveDataUnavailable(f"cannot read signal SHM context for {context_date}: {exc}") from exc
        series = self._series_for_context(signal, universe)
        try:
            self._require_complete_signal_prefix(ctx, i)
            ctx.write_signal(i, series)
        except LiveDataUnavailable:
            self._close_context(context_date)
            raise
        except Exception as exc:
            self._close_context(context_date)
            raise LiveDataUnavailable(f"cannot write signal SHM context for {context_date}: {exc}") from exc
        return f"sdk://signal/{context_date}/{i}"


@dataclass
class CachedCheckpointModel:
    info: CheckpointInfo
    model: Any
    feature_std: np.ndarray
    feature_clip: float
    label_std: np.ndarray
    label_horizons: list[int]
    use_vola: bool
    prepared_target_time: pd.Timestamp | None = None
    prepared_history_end_time: pd.Timestamp | None = None
    prepared_columns: list[str] | None = None
    prepared_feature_window: FeatureWindow | None = None
    cache_consumed: bool = True
    last_cache_hit: bool = False
    last_rebuild_reason: str | None = None
    last_prepared_steps: int = 0


@dataclass(frozen=True)
class SignalOnlyRuntime:
    infos: list[CheckpointInfo]
    consistency: dict[str, Any]
    feature_plan: FeaturePlan
    writer: AuditSignalWriter
    cached_models: list[CachedCheckpointModel]
    data_source: Any | None = None


def read_universe_symbols(date: str, universe_name: str | None) -> list[str] | None:
    if not universe_name:
        return None

    try:
        universe = dbtool.read_wide(date, f"source/meta/{universe_name}").squeeze()
    except Exception as exc:
        raise LiveDataUnavailable(f"cannot read universe {universe_name!r} for {date}: {exc}") from exc
    return [str(symbol) for symbol in universe.index[universe != 0]]


def _unique_dates(dates: list[str]) -> list[str]:
    seen = set()
    result = []
    for date in dates:
        if date not in seen:
            seen.add(date)
            result.append(date)
    return result


def _read_1h_shm_probe_index(shm: Any, path: Path, alpha: str = "ret1m") -> pd.DatetimeIndex:
    store = shm.from_path(path, readonly=True)
    try:
        frame = store.read(f"feature/{alpha}")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"feature/{alpha} is not a DataFrame: {type(frame)}")
    if frame.empty:
        raise ValueError(f"feature/{alpha} is empty")
    return _normalize_index(frame.index)


def resolve_1h_shm_date(task_time: str | None) -> str:
    task_ts = _parse_task_timestamp(task_time) or pd.Timestamp.now(tz="Asia/Shanghai").floor("h")
    natural_date = task_ts.strftime("%Y-%m-%d")
    # shm_root, metadata, shm
    candidate_dates = [natural_date]
    try:
        candidate_dates.append(str(metadata.prev_date(natural_date)))
    except Exception:
        candidate_dates.append((task_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

    errors = []
    for date in _unique_dates(candidate_dates):
        path = Path(SHM_ROOT) / date / DEFAULT_SDK_FREQ
        if not path.exists():
            errors.append(f"{date}: missing {path}")
            continue
        try:
            index = _normalize_index(metadata.index(date))
        except Exception as exc:
            try:
                index = _read_1h_shm_probe_index(shm, path)
            except Exception as fallback_exc:
                errors.append(
                    f"{date}: cannot read metadata index: {exc}; "
                    f"cannot read feature/ret1m index fallback: {fallback_exc}"
                )
                continue
        if bool((index == task_ts).any()):
            return date
        errors.append(f"{date}: index does not contain {task_ts}")

    raise LiveDataUnavailable(f"cannot resolve 1h SHM date for {task_ts}; " + "; ".join(errors))


def _filter_feature_frame_to_universe(frame: pd.DataFrame, universe: list[str], alpha: str) -> pd.DataFrame:
    copied = frame.copy()
    copied.columns = pd.Index([str(column) for column in copied.columns])
    if not copied.columns.is_unique:
        raise LiveDataUnavailable(f"feature/{alpha} columns are not unique after string conversion")
    missing = [symbol for symbol in universe if symbol not in set(copied.columns)]
    if missing:
        raise LiveDataUnavailable(f"feature/{alpha} missing universe symbols: {missing[:8]}")
    return copied.loc[:, universe]


def filter_feature_frames_to_universe(frames: dict[str, pd.DataFrame], universe: list[str] | None) -> dict[str, pd.DataFrame]:
    if universe is None:
        return frames
    return {alpha: _filter_feature_frame_to_universe(frame, universe, alpha) for alpha, frame in frames.items()}


def read_1h_shm_features(date: str, alphas: list[str]) -> dict[str, pd.DataFrame]:
    # SHM_ROOT, _metadata, shm = _load_mona_shm_modules()

    path = SHM_ROOT / date / "1h"
    if not path.exists():
        raise LiveDataUnavailable(f"1h SHM path does not exist: {path}")
    try:
        store = shm.from_path(path, readonly=True)
    except Exception as exc:
        raise LiveDataUnavailable(f"cannot open 1h SHM {path}: {exc}") from exc
    try:
        frames = {}
        for alpha in alphas:
            key = f"feature/{alpha}"
            try:
                frames[alpha] = store.read(key)
            except Exception as exc:
                raise LiveDataUnavailable(f"cannot read {key} from {path}: {exc}") from exc
        return frames
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def _read_1h_shm_feature_step_from_store(
    store: Any, path: Path, alphas: list[str], task_time: str) -> dict[str, pd.DataFrame]:

    target = _parse_task_timestamp(task_time)
    if target is None:
        raise ValueError("task_time is required for feature step loading")

    frames = {}
    for alpha in alphas:
        key = f"feature/{alpha}"
        try:
            data = store.data(key)
            version = store.version(key)
            data_map = store.get_map(f"{key}-data")
        except Exception as exc:
            raise ShmAccessUnavailable(f"cannot read {key} from {path}: {exc}") from exc

        n = int(np.searchsorted(version == -1, True))
        raw_index = data_map.meta.index
        if raw_index is None:
            index = pd.to_datetime(version[:n], unit="ns", utc=True).tz_convert("Asia/Shanghai")
        else:
            index = _normalize_index(raw_index[:n])
        positions = np.flatnonzero(index == target)
        if len(positions) != 1:
            latest = index[-1] if len(index) else None
            raise LiveDataUnavailable(f"exact target {target} is not available for {key}; latest={latest}")
        pos = int(positions[0])
        columns = [str(col) for col in data_map.meta.columns]
        row = np.asarray(data[pos], dtype=np.float32)[None, :].copy()
        frames[alpha] = pd.DataFrame(row, index=pd.DatetimeIndex([index[pos]]), columns=columns)
    return frames


def _read_1h_shm_feature_step_window_from_store(
    store: Any, path: Path, feature_plan: FeaturePlan, task_time: str, *, source_date: str | None = None) -> FeatureWindow:

    target = _parse_task_timestamp(task_time)
    if target is None:
        raise ValueError("task_time is required for feature step loading")

    reference_columns: list[str] | None = None
    reference_index: pd.DatetimeIndex | None = None
    reference_pos: int | None = None
    feature_lengths: dict[str, int] = {}
    values: np.ndarray | None = None
    for channel, alpha in enumerate(feature_plan.alphas):
        key = f"feature/{alpha}"
        try:
            data = store.data(key)
            version = store.version(key)
            data_map = store.get_map(f"{key}-data")
        except Exception as exc:
            raise ShmAccessUnavailable(f"cannot read {key} from {path}: {exc}") from exc

        n = int(np.searchsorted(version == -1, True))
        raw_index = data_map.meta.index
        if raw_index is None:
            index = pd.to_datetime(version[:n], unit="ns", utc=True).tz_convert("Asia/Shanghai")
        else:
            index = _normalize_index(raw_index[:n])
        columns = [str(col) for col in data_map.meta.columns]
        feature_lengths[alpha] = n

        if reference_columns is None:
            reference_columns = columns
            positions = np.flatnonzero(index == target)
            if len(positions) != 1:
                latest = index[-1] if len(index) else None
                raise LiveDataUnavailable(f"exact target {target} is not available for {key}; latest={latest}")
            reference_pos = int(positions[0])
            reference_index = pd.DatetimeIndex([index[reference_pos]])
            values = np.empty((len(columns), len(feature_plan.alphas), 1), dtype=np.float32)
        elif columns != reference_columns:
            raise ValueError(f"columns mismatch for feature/{alpha}")
        else:
            assert reference_pos is not None
            if len(index) <= reference_pos or pd.Timestamp(index[reference_pos]) != target:
                raise LiveDataUnavailable(f"exact target {target} is not aligned for {key}")

        assert values is not None and reference_pos is not None
        values[:, channel, 0] = np.asarray(data[reference_pos], dtype=np.float32)

    assert values is not None and reference_columns is not None and reference_index is not None
    return FeatureWindow(
        values=values,
        columns=reference_columns,
        index=reference_index,
        selected_time=target,
        requested_time=target,
        task_time_floor_used=False,
        feature_lengths=feature_lengths,
        source_date=source_date,
    )


def filter_feature_window_to_universe(feature_window: FeatureWindow, universe: list[str] | None) -> FeatureWindow:
    if universe is None:
        return feature_window
    columns = [str(column) for column in feature_window.columns]
    if len(columns) != len(set(columns)):
        raise LiveDataUnavailable("feature window columns are not unique after string conversion")
    position_by_symbol = {symbol: i for i, symbol in enumerate(columns)}
    missing = [symbol for symbol in universe if symbol not in position_by_symbol]
    if missing:
        raise LiveDataUnavailable(f"feature window missing universe symbols: {missing[:8]}")
    positions = [position_by_symbol[symbol] for symbol in universe]
    return FeatureWindow(
        values=feature_window.values[positions, :, :].copy(), columns=list(universe), index=feature_window.index,
        selected_time=feature_window.selected_time, requested_time=feature_window.requested_time,
        task_time_floor_used=feature_window.task_time_floor_used, feature_lengths=feature_window.feature_lengths,
        source_date=feature_window.source_date,
    )


class ShmDataSource:
    source_name = "shm"

    def __init__(self, universe_name: str | None = None) -> None:
        self.universe_name = universe_name
        self._cached_universe_date: str | None = None
        self._cached_universe_symbols: list[str] | None = None
        self._cached_date: str | None = None
        self._cached_alphas: tuple[str, ...] | None = None
        self._cached_frame_universe_date: str | None = None
        self._cached_frames: dict[str, pd.DataFrame] | None = None
        self._cached_index: pd.DatetimeIndex | None = None
        self._force_refresh = True
        self._step_store_date: str | None = None
        self._step_store_alphas: tuple[str, ...] | None = None
        self._step_store: Any | None = None

    def _universe(self, date: str) -> list[str] | None:
        if not self.universe_name:
            return None
        if self._cached_universe_date != date:
            symbols = read_universe_symbols(date, self.universe_name)
            if not symbols:
                raise LiveDataUnavailable(f"strategy universe is empty for {self.universe_name!r} on {date}")
            self._cached_universe_symbols = symbols
            self._cached_universe_date = date
        return self._cached_universe_symbols

    def _cache_covers(self, task_time: str | None) -> bool:
        if self._cached_date is None or self._cached_index is None or self._cached_index.empty:
            return False
        task_ts = _parse_task_timestamp(task_time)
        if task_ts is None:
            return False
        return bool(self._cached_index[0] <= task_ts <= self._cached_index[-1])

    def resolve_date(self, task_time: str | None) -> str:
        if self._cache_covers(task_time):
            self._force_refresh = False
            assert self._cached_date is not None
            return self._cached_date
        self._force_refresh = True
        return resolve_1h_shm_date(task_time)

    def invalidate_feature_frame_cache(self) -> None:
        self._cached_frames = None
        self._cached_index = None
        self._force_refresh = True

    def load_feature_frames(self, date: str, alphas: list[str]) -> dict[str, pd.DataFrame]:
        return self.load_feature_frames_for_universe(date, alphas, date)

    def load_feature_frames_for_universe(self, date: str, alphas: list[str], universe_date: str) -> dict[str, pd.DataFrame]:
        alpha_key = tuple(alphas)
        universe_key = str(universe_date)
        if (
            not self._force_refresh
            and self._cached_date == date
            and self._cached_alphas == alpha_key
            and self._cached_frame_universe_date == universe_key
            and self._cached_frames is not None
        ):
            return self._cached_frames

        frames = filter_feature_frames_to_universe(read_1h_shm_features(date, alphas), self._universe(universe_key))
        first = frames[alphas[0]]
        self._cached_date = date
        self._cached_alphas = alpha_key
        self._cached_frame_universe_date = universe_key
        self._cached_frames = frames
        self._cached_index = _normalize_index(first.index)
        self._force_refresh = False
        return frames

    def prepare_feature_step_reader(self, date: str, alphas: list[str]) -> None:
        alpha_key = tuple(alphas)
        if self._step_store is not None and self._step_store_date == date and self._step_store_alphas == alpha_key:
            return
        self.invalidate_feature_step_reader()
        # SHM_ROOT, _metadata, shm = _load_f2_shm_modules()
        path = SHM_ROOT / date / "1h"
        if not path.exists():
            raise LiveDataUnavailable(f"1h SHM path does not exist: {path}")
        store = None
        try:
            store = shm.from_path(path, readonly=True)
            for alpha in alphas:
                key = f"feature/{alpha}"
                store.data(key)
                store.version(key)
                store.get_map(f"{key}-data")
        except Exception as exc:
            close = getattr(store, "close", None)
            if callable(close):
                close()
            raise ShmAccessUnavailable(f"cannot prepare feature step reader for {path}: {exc}") from exc
        self._step_store = store
        self._step_store_date = date
        self._step_store_alphas = alpha_key

    def invalidate_feature_step_reader(self) -> None:
        store = self._step_store
        self._step_store = None
        self._step_store_date = None
        self._step_store_alphas = None
        close = getattr(store, "close", None)
        if callable(close):
            close()

    def load_feature_step_frames(self, date: str, alphas: list[str], task_time: str) -> dict[str, pd.DataFrame]:
        try:
            self.prepare_feature_step_reader(date, alphas)
            assert self._step_store is not None
            # SHM_ROOT, _metadata, _shm = _load_f2_shm_modules()
            frames = _read_1h_shm_feature_step_from_store(self._step_store, SHM_ROOT / date / "1h", alphas, task_time)
            return filter_feature_frames_to_universe(frames, self._universe(date))
        except ShmAccessUnavailable:
            self.invalidate_feature_step_reader()
            raise
        except LiveDataUnavailable:
            raise
        except ValueError:
            raise
        except Exception as exc:
            self.invalidate_feature_step_reader()
            raise ShmAccessUnavailable(f"cannot read feature step for {date}: {exc}") from exc

    def load_feature_step_window(self, date: str, feature_plan: FeaturePlan, task_time: str) -> FeatureWindow:
        try:
            self.prepare_feature_step_reader(date, feature_plan.alphas)
            assert self._step_store is not None
            # SHM_ROOT, _metadata, _shm = _load_f2_shm_modules()
            feature_window = _read_1h_shm_feature_step_window_from_store(
                self._step_store, SHM_ROOT / date / "1h", feature_plan, task_time, source_date=date)
            return filter_feature_window_to_universe(feature_window, self._universe(date))
        except ShmAccessUnavailable:
            self.invalidate_feature_step_reader()
            raise
        except LiveDataUnavailable:
            raise
        except ValueError:
            raise
        except Exception as exc:
            self.invalidate_feature_step_reader()
            raise ShmAccessUnavailable(f"cannot read feature step window for {date}: {exc}") from exc


class DumpDataSource:
    source_name = "dump"

    def __init__(self, dump_dir: str | Path) -> None:
        self.dump_dir = Path(dump_dir)

    def load_feature_frames(self, date: str, alphas: list[str]) -> dict[str, pd.DataFrame]:
        metadata_path = self.dump_dir / "metadata.json"
        if not metadata_path.exists():
            raise LiveDataUnavailable(f"dump metadata does not exist: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_files = metadata.get("frame_files") or {}
        missing = [alpha for alpha in alphas if alpha not in frame_files]
        if missing:
            raise LiveDataUnavailable(f"dump missing feature frames: {missing}")
        frames = {}
        for alpha in alphas:
            path = self.dump_dir / frame_files[alpha]
            if not path.exists():
                raise LiveDataUnavailable(f"dump feature file does not exist for {alpha}: {path}")
            frames[alpha] = pd.read_pickle(path)
        return frames


def make_writer(mode: str, output_dir: str | Path, allow_unsafe_real_write: bool = False) -> Any:
    if mode == "dry-run":
        return AuditSignalWriter(output_dir)
    if mode == "sdk":
        return SDKSignalWriter()
    if mode != "dry-run":
        if not allow_unsafe_real_write:
            raise UnsafeRealWriteError("real DB signal write is disabled; use --writer sdk for f2.sdk SHM signal write")
        raise UnsafeRealWriteError("real DB signal write is intentionally not implemented in this signal-only entry")
    raise ValueError(f"unsupported writer mode: {mode!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OKX 1h signal-only entry")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--state-dict-key", default="ema_state_dict")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--signal-key", default=DEFAULT_SIGNAL_KEY)
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--writer", choices=["dry-run", "sdk"], default="dry-run")
    parser.add_argument("--allow-unsafe-real-write", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=3600.0)
    parser.add_argument("--task-time", default=None)
    parser.add_argument("--source", choices=["shm", "dump"], default="shm")
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--dump-feature-frames", default=None)
    parser.add_argument("--wait-timeout-seconds", type=float, default=None)
    parser.add_argument("--wait-poll-seconds", type=float, default=1.0)
    parser.add_argument("--fake-signal", action="store_true", help="write an explicit fake zero signal audit; never DB")
    parser.add_argument("--fake-universe-file", default=None)
    parser.add_argument(
        "--debug-replay-duplicates",
        action="store_true",
        help="dry-run loop helper: write repeated local audits for the same selected_time",
    )
    parser.add_argument("--inspect-report", default=None)
    return parser.parse_args()


def _parse_task_timestamp(task_time: str | None) -> pd.Timestamp | None:
    if task_time is None:
        return None
    ts = pd.Timestamp(task_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Shanghai")
    else:
        ts = ts.tz_convert("Asia/Shanghai")
    return ts


def load_last_audit_selected_time(audit_dir: str | Path, signal_key: str) -> pd.Timestamp | None:
    root = Path(audit_dir) / signal_key
    if not root.exists():
        return None
    audit_paths = sorted(root.glob("*/*/audit.json"))
    for path in reversed(audit_paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        selected_time = payload.get("selected_time")
        if selected_time:
            return _parse_task_timestamp(str(selected_time))
    return None


def initial_loop_selected_time(args: argparse.Namespace) -> pd.Timestamp | None:
    if not getattr(args, "loop", False) or getattr(args, "writer", None) == "sdk":
        return None
    return load_last_audit_selected_time(args.audit_dir, args.signal_key)


def discover_versions(model_dir: str | Path) -> list[Path]:
    root = Path(model_dir)
    versions = sorted(p for p in root.iterdir() if p.is_dir() and (p / "9.pth.tar").exists())
    return versions


def build_feature_plan(config: dict[str, Any]) -> FeaturePlan:
    dataset = config.get("dataset") or {}
    model = config.get("model") or {}
    alphas = list(dataset.get("alphas") or [])
    if not alphas:
        raise ValueError("checkpoint config missing dataset.alphas")
    in_channels = int(model.get("in_channels", len(alphas)))
    if in_channels != len(alphas):
        raise ValueError(f"model.in_channels {in_channels} != len(dataset.alphas) {len(alphas)}")
    return FeaturePlan(
        alphas=alphas,
        seq_len=int(dataset["seq_len"]),
        in_channels=in_channels,
        out_channels=int(model["out_channels"]),
        label_name=dataset.get("label_name"),
        model_name=str(model.get("name")),
    )


def _summarize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    keys = list(state_dict.keys())
    tensors = []
    total_params = 0
    for key in keys:
        value = state_dict[key]
        shape = list(getattr(value, "shape", []))
        numel = int(value.numel()) if hasattr(value, "numel") else None
        if numel is not None:
            total_params += numel
        tensors.append({"key": key, "shape": shape, "dtype": str(getattr(value, "dtype", "")), "numel": numel})
    return {
        "key_count": len(keys),
        "first_keys": keys[:12],
        "last_keys": keys[-8:],
        "total_params": total_params,
        "tensor_sample": tensors[:16],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sample": value.reshape(-1)[:8].tolist(),
        }
    if hasattr(value, "__dict__"):
        return {"type": f"{value.__class__.__module__}.{value.__class__.__name__}", **_jsonable(value.__dict__)}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _summarize_normalizer(normalizer: Any) -> dict[str, Any]:
    if not isinstance(normalizer, dict):
        return {"type": str(type(normalizer)), "value": _jsonable(normalizer)}
    summary = {}
    for key, value in normalizer.items():
        entry: dict[str, Any] = {"type": f"{value.__class__.__module__}.{value.__class__.__name__}"}
        if hasattr(value, "std"):
            std = np.asarray(value.std)
            entry.update(
                {
                    "std_shape": list(std.shape),
                    "std_dtype": str(std.dtype),
                    "std_min": float(np.nanmin(std)),
                    "std_max": float(np.nanmax(std)),
                    "std_sample": std.reshape(-1)[:8].tolist(),
                }
            )
        if hasattr(value, "clip"):
            entry["clip"] = _jsonable(value.clip)
        summary[str(key)] = entry
    return summary


def _confirm_output_semantics(config: dict[str, Any], checkpoint: dict[str, Any], state_dict_key: str) -> dict[str, Any]:
    model_config = copy.deepcopy(config["model"])
    model = gen_model(**model_config, seq_len=config["dataset"]["seq_len"])
    model.load_state_dict(checkpoint[state_dict_key])
    model.eval()
    feature_count = len(config["dataset"]["alphas"])
    seq_len = int(config["dataset"]["seq_len"])
    with torch.no_grad():
        x = torch.zeros(1, 3, feature_count, 1)
        if hasattr(model, "reset_eval_model"):
            model.reset_eval_model("incremental_eval")
        if hasattr(model, "reset_cache_model"):
            model.reset_cache_model()
        pred = model(x)
    if isinstance(pred, tuple):
        pred_shape = list(pred[0].shape)
        prob_shape = list(pred[1].shape)
        returns_prob = True
    else:
        pred_shape = list(pred.shape)
        prob_shape = None
        returns_prob = False
    return {
        "model_class": model.__class__.__name__,
        "input_shape_checked": [1, 3, feature_count, 1],
        "seq_len": seq_len,
        "returns_prob": returns_prob,
        "pred_shape": pred_shape,
        "prob_shape": prob_shape,
        "out_channels": int(config["model"]["out_channels"]),
        "channel_semantics": "confirmed single output channel" if int(config["model"]["out_channels"]) == 1 else "multi-channel output; requires explicit channel selection",
    }


def inspect_checkpoints(model_dir: str | Path, state_dict_key: str = "ema_state_dict") -> list[CheckpointInfo]:
    infos = []
    for version_dir in discover_versions(model_dir):
        path = version_dir / "9.pth.tar"
        checkpoint = _torch_load_checkpoint(path)
        config = checkpoint["config"]
        feature_plan = build_feature_plan(config)
        if state_dict_key not in checkpoint:
            raise KeyError(f"{path} missing {state_dict_key}")
        infos.append(
            CheckpointInfo(
                version=version_dir.name,
                path=str(path),
                load_ok=True,
                top_level_keys=list(checkpoint.keys()),
                config=config,
                feature_plan=feature_plan,
                state_dict_key=state_dict_key,
                state_dict_summary=_summarize_state_dict(checkpoint[state_dict_key]),
                normalizer_summary=_summarize_normalizer(checkpoint.get("normalizer")),
                output_semantics=_confirm_output_semantics(config, checkpoint, state_dict_key),
            )
        )
    return infos


def checkpoint_consistency(infos: list[CheckpointInfo]) -> dict[str, Any]:
    if not infos:
        return {"version_count": 0, "consistent": False, "reason": "no checkpoints"}
    first = infos[0]
    fields = {
        "alphas": [info.feature_plan.alphas for info in infos],
        "seq_len": [info.feature_plan.seq_len for info in infos],
        "in_channels": [info.feature_plan.in_channels for info in infos],
        "out_channels": [info.feature_plan.out_channels for info in infos],
        "label_name": [info.feature_plan.label_name for info in infos],
        "model_name": [info.feature_plan.model_name for info in infos],
        "state_dict_keys": [info.state_dict_summary["first_keys"] + info.state_dict_summary["last_keys"] for info in infos],
    }
    equal = {name: all(value == values[0] for value in values) for name, values in fields.items()}
    return {
        "version_count": len(infos),
        "versions": [info.version for info in infos],
        "consistent": all(equal.values()),
        "checks": equal,
        "reference": {
            "alphas_count": len(first.feature_plan.alphas),
            "seq_len": first.feature_plan.seq_len,
            "in_channels": first.feature_plan.in_channels,
            "out_channels": first.feature_plan.out_channels,
            "label_name": first.feature_plan.label_name,
            "model_name": first.feature_plan.model_name,
        },
    }


def make_writer(mode: str, output_dir: str | Path, allow_unsafe_real_write: bool = False) -> Any:
    if mode == "dry-run":
        return AuditSignalWriter(output_dir)
    if mode == "sdk":
        return SDKSignalWriter()
    if mode != "dry-run":
        if not allow_unsafe_real_write:
            raise UnsafeRealWriteError("real DB signal write is disabled; use --writer sdk for f2.sdk SHM signal write")
        raise UnsafeRealWriteError("real DB signal write is intentionally not implemented in this signal-only entry")
    raise ValueError(f"unsupported writer mode: {mode!r}")


def make_data_source(args: argparse.Namespace) -> Any:
    source = getattr(args, "source", "shm")
    if source == "shm":
        return ShmDataSource(universe_name=getattr(args, "universe", None))
    if source == "dump":
        dump_dir = getattr(args, "dump_dir", None)
        if not dump_dir:
            raise ValueError("--dump-dir is required when --source=dump")
        return DumpDataSource(dump_dir)
    raise ValueError(f"unsupported --source={source!r}")


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _reset_model_for_incremental_inference(model: Any) -> None:
    if hasattr(model, "reset_eval_model"):
        model.reset_eval_model("incremental_eval")
    else:
        raise ValueError("model does not support incremental_eval")
    if hasattr(model, "reset_cache_model"):
        model.reset_cache_model()


def _normalizer_feature_std(normalizer: dict[str, Any]) -> np.ndarray:
    std = np.asarray(normalizer["feature"].std, dtype=np.float32)
    if std.ndim != 1:
        raise ValueError(f"feature normalizer std must be 1-D, got {std.shape}")
    return std


def label_horizons(label_name: Any, out_channels: int) -> list[int]:
    if isinstance(label_name, str):
        names = [label_name]
    elif isinstance(label_name, (list, tuple)):
        names = list(label_name)
    else:
        raise ValueError(f"checkpoint dataset.label_name must be str/list/tuple, got {type(label_name)}")
    if len(names) != out_channels:
        raise ValueError(f"label_name count {len(names)} must equal model out_channels {out_channels}")
    horizons = []
    for name in names:
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        if not digits:
            raise ValueError(f"label_name must include digits for vola horizon: {name!r}")
        horizons.append(int(digits))
    return horizons


def load_cached_checkpoint_models(infos: list[CheckpointInfo], state_dict_key: str) -> list[CachedCheckpointModel]:
    # os.environ.setdefault("EXCHANGE", "okx10m")

    torch.set_num_threads(1)
    cached = []
    for info in infos:
        checkpoint = _torch_load_checkpoint(Path(info.path))
        config = checkpoint["config"]
        normalizer = checkpoint["normalizer"]
        model = gen_model(**copy.deepcopy(config["model"]), seq_len=config["dataset"]["seq_len"])
        model.load_state_dict(checkpoint[state_dict_key])
        model.eval()
        _reset_model_for_incremental_inference(model)
        cached.append(
            CachedCheckpointModel(
                info=info,
                model=model,
                feature_std=_normalizer_feature_std(normalizer),
                feature_clip=float(normalizer["feature"].clip),
                label_std=np.asarray(normalizer["label"].std, dtype=np.float32),
                label_horizons=label_horizons(config["dataset"].get("label_name"), int(config["model"]["out_channels"])),
                use_vola=bool(config["dataset"].get("use_vola", True)),
            )
        )
    return cached


def prepare_signal_runtime(args: argparse.Namespace) -> SignalOnlyRuntime:
    infos = inspect_checkpoints(args.model_dir, args.state_dict_key)
    if len(infos) != 8:
        raise ValueError(f"expected exactly 8 checkpoint versions, got {len(infos)}")
    consistency = checkpoint_consistency(infos)
    if not consistency["consistent"]:
        raise ValueError(f"checkpoint configs are inconsistent: {consistency}")

    feature_plan = infos[0].feature_plan
    writer = make_writer(args.writer, args.audit_dir, args.allow_unsafe_real_write)
    data_source = None if args.fake_signal else make_data_source(args)
    cached_models = [] if args.fake_signal else load_cached_checkpoint_models(infos, args.state_dict_key)

    return SignalOnlyRuntime(
        infos=infos, consistency=consistency, feature_plan=feature_plan, writer=writer, cached_models=cached_models, data_source=data_source)


def _prepared_runtime_history_window(
    cached_models: list[CachedCheckpointModel], target_time: pd.Timestamp) -> FeatureWindow | None:

    if not cached_models:
        return None
    first_history = cached_models[0].prepared_feature_window
    if first_history is None:
        return None
    for cached in cached_models:
        if cached.cache_consumed:
            return None
        if cached.prepared_target_time is None or pd.Timestamp(cached.prepared_target_time) != pd.Timestamp(target_time):
            return None
        if cached.prepared_history_end_time is None or pd.Timestamp(cached.prepared_history_end_time) != pd.Timestamp(first_history.selected_time):
            return None
        if cached.prepared_columns != list(first_history.columns):
            return None
        if cached.prepared_feature_window is None:
            return None
    return first_history


def _format_task_timestamp(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")


def _resolve_feature_date(data_source: Any, task_time: str) -> str:
    resolve_date = getattr(data_source, "resolve_date", None)
    if callable(resolve_date):
        return str(resolve_date(task_time))
    return pd.Timestamp(task_time).strftime("%Y-%m-%d")


def _history_tail_window_for_target(
    feature_window: FeatureWindow,
    feature_plan: FeaturePlan,
    target_time: pd.Timestamp,
) -> FeatureWindow:
    history_len = feature_plan.seq_len - 1
    if history_len < 1:
        raise ValueError(f"seq_len must be at least 2 for prewarm, got {feature_plan.seq_len}")
    if feature_window.values.shape[-1] < history_len:
        raise ValueError(
            f"current feature window has {feature_window.values.shape[-1]} rows, "
            f"need {history_len} rows for next-target prewarm"
        )
    target = pd.Timestamp(target_time)
    history_index = pd.DatetimeIndex(feature_window.index[-history_len:])
    if pd.Timestamp(history_index[-1]) != _expected_history_end_for_target(target):
        raise ValueError("current feature window does not end at next_target-1h")
    return FeatureWindow(
        values=feature_window.values[:, :, -history_len:].copy(),
        columns=list(feature_window.columns),
        index=history_index,
        selected_time=history_index[-1],
        requested_time=history_index[-1],
        task_time_floor_used=False,
        feature_lengths=feature_window.feature_lengths,
        source_date=feature_window.source_date,
    )


def _parallel_map_cached_models(cached_models: list[Any], feature_window: FeatureWindow, infer_func: Any) -> list[Any]:
    if len(cached_models) <= 1:
        return [infer_func(cached, feature_window) for cached in cached_models]
    max_workers = min(len(cached_models), max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda cached: infer_func(cached, feature_window), cached_models))


def load_prewarm_feature_window(
    data_source: Any,
    task_time: str,
    feature_plan: FeaturePlan,
    *,
    target_feature_date: str | None = None,
    wait_timeout_seconds: float | None = None,
    wait_poll_seconds: float = 1.0,
) -> FeatureWindow:
    target = _parse_task_timestamp(task_time)
    if target is None:
        raise ValueError("task_time is required for prewarm feature loading")
    history_end = target - pd.Timedelta(hours=1)
    history_plan = _feature_plan_with_seq_len(feature_plan, feature_plan.seq_len - 1)
    return load_feature_window(
        data_source,
        _format_task_timestamp(history_end),
        history_plan,
        exact_task_time=True,
        feature_date_override=(
            target_feature_date
            if callable(getattr(data_source, "load_feature_frames_for_universe", None))
            else None
        ),
        universe_date=target_feature_date,
        wait_timeout_seconds=wait_timeout_seconds,
        wait_poll_seconds=wait_poll_seconds,
    )


def prepare_target_cache(
    runtime: SignalOnlyRuntime, data_source: Any,
    target_time: pd.Timestamp, *, current_feature_window: FeatureWindow | None = None,
    wait_timeout_seconds: float | None = None, wait_poll_seconds: float = 1.0) -> bool:

    if not _supports_prepared_runtime(runtime.cached_models):
        return False
    target = pd.Timestamp(target_time)
    if _prepared_runtime_history_window(runtime.cached_models, target) is not None:
        return True

    feature_plan = runtime.feature_plan
    task_time = _format_task_timestamp(target)
    target_feature_date = _resolve_feature_date(data_source, task_time)
    can_filter_history_to_target_universe = callable(getattr(data_source, "load_feature_frames_for_universe", None))
    can_reuse_current_window = (
        current_feature_window is not None
        and (
            not can_filter_history_to_target_universe
            or current_feature_window.source_date is None
            or str(current_feature_window.source_date) == target_feature_date
        )
    )

    if can_reuse_current_window:
        assert current_feature_window is not None
        history_window = _history_tail_window_for_target(current_feature_window, feature_plan, target)
    else:
        history_window = load_prewarm_feature_window(
            data_source, task_time, feature_plan, target_feature_date=target_feature_date,
            wait_timeout_seconds=wait_timeout_seconds, wait_poll_seconds=wait_poll_seconds,
        )

    _parallel_map_cached_models(
        runtime.cached_models, history_window, lambda cached, window: prewarm_cached_checkpoint_for_target(cached, window, target))

    step_prepare = getattr(data_source, "prepare_feature_step_reader", None)
    if callable(step_prepare):
        step_prepare(target_feature_date, feature_plan.alphas)
    return True


def prepare_next_target_cache(
    runtime: SignalOnlyRuntime, data_source: Any, current_selected_time: pd.Timestamp, *,
    current_feature_window: FeatureWindow | None = None, wait_timeout_seconds: float | None = None, wait_poll_seconds: float = 1.0,
) -> bool:
    next_target = pd.Timestamp(current_selected_time) + pd.Timedelta(hours=1)
    return prepare_target_cache(
        runtime, data_source, next_target, current_feature_window=current_feature_window,
        wait_timeout_seconds=wait_timeout_seconds, wait_poll_seconds=wait_poll_seconds,
    )


def _load_universe_for_fake(path: str | None) -> list[str]:
    if not path:
        return ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"empty universe file: {path}")
    return values


def _fake_signal(universe: list[str], task_time: str) -> pd.DataFrame:
    index = [pd.Timestamp(task_time, tz="Asia/Shanghai")]
    return pd.DataFrame(np.zeros((1, len(universe)), dtype=float), index=index, columns=universe)


def _supports_prepared_runtime(cached_models: list[Any]) -> bool:
    return bool(cached_models) and all(isinstance(cached, CachedCheckpointModel) for cached in cached_models)


def _expected_history_end_for_target(target_time: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(target_time) - pd.Timedelta(hours=1)


def _feature_window_history_slice(feature_window: FeatureWindow, target_time: pd.Timestamp) -> tuple[slice, pd.Timestamp]:
    target = pd.Timestamp(target_time)
    if len(feature_window.index) < 2:
        raise ValueError("prewarm requires at least one history row and one target row")
    selected_time = pd.Timestamp(feature_window.selected_time)
    if selected_time == target:
        return slice(None, -1), pd.Timestamp(feature_window.index[-2])
    history_end = pd.Timestamp(feature_window.index[-1])
    if history_end == _expected_history_end_for_target(target):
        return slice(None), history_end
    raise ValueError(f"feature window does not contain history ending at {target - pd.Timedelta(hours=1)}")


def _normalize_cached_feature(
    cached: CachedCheckpointModel, feature_window: FeatureWindow, time_slice: slice = slice(None)) -> np.ndarray:

    if len(cached.feature_std) != feature_window.values.shape[1]:
        raise ValueError(f"feature std length {len(cached.feature_std)} != input channels {feature_window.values.shape[1]}")
    values = feature_window.values[:, :, time_slice]
    norm_feature = values / cached.feature_std[None, :, None]
    norm_feature = np.clip(norm_feature, -cached.feature_clip, cached.feature_clip)
    norm_feature[~np.isfinite(norm_feature)] = 0.0
    return np.ascontiguousarray(norm_feature, dtype=np.float32)


def _forward_cached_model(cached: CachedCheckpointModel, norm_feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.inference_mode():
        pred_out = cached.model(torch.from_numpy(norm_feature)[None, ...])
    if not isinstance(pred_out, tuple):
        raise ValueError("expected model to return (pred, prob)")
    pred_tensor, prob_tensor = pred_out
    pred = pred_tensor.detach().cpu().numpy()[0, :, 0, -1]
    prob = prob_tensor.detach().cpu().sigmoid().numpy()[0, :, 0, -1] * 2 - 1
    return pred, prob


def prewarm_cached_checkpoint_for_target(
    cached: CachedCheckpointModel, feature_window: FeatureWindow, target_time: pd.Timestamp) -> None:

    history_slice, history_end = _feature_window_history_slice(feature_window, pd.Timestamp(target_time))
    norm_feature = _normalize_cached_feature(cached, feature_window, history_slice)
    if norm_feature.shape[-1] < 1:
        raise ValueError("prewarm requires non-empty history")

    _reset_model_for_incremental_inference(cached.model)
    for step in range(norm_feature.shape[-1]):
        _forward_cached_model(cached, norm_feature[:, :, step : step + 1])

    cached.prepared_target_time = pd.Timestamp(target_time)
    cached.prepared_history_end_time = history_end
    cached.prepared_columns = list(feature_window.columns)
    cached.prepared_feature_window = FeatureWindow(
        values=feature_window.values[:, :, history_slice].copy(),
        columns=list(feature_window.columns),
        index=pd.DatetimeIndex(feature_window.index[history_slice]),
        selected_time=history_end,
        requested_time=history_end,
        task_time_floor_used=False,
        feature_lengths=feature_window.feature_lengths,
        source_date=feature_window.source_date,
    )
    cached.cache_consumed = False
    cached.last_cache_hit = False
    cached.last_rebuild_reason = None
    cached.last_prepared_steps = int(norm_feature.shape[-1])


def _feature_plan_with_seq_len(feature_plan: FeaturePlan, seq_len: int) -> FeaturePlan:
    return FeaturePlan(
        alphas=feature_plan.alphas,
        seq_len=seq_len,
        in_channels=feature_plan.in_channels,
        out_channels=feature_plan.out_channels,
        label_name=feature_plan.label_name,
        model_name=feature_plan.model_name,
    )

def _load_feature_step_frames(
    data_source: Any, date: str, alphas: list[str], task_time: str) -> dict[str, pd.DataFrame]:

    loader = getattr(data_source, "load_feature_step_frames", None)
    if callable(loader):
        return loader(date, alphas, task_time)
    frames = data_source.load_feature_frames(date, alphas)
    target = _parse_task_timestamp(task_time)
    if target is None:
        raise ValueError("task_time is required for feature step loading")
    result = {}
    for alpha, frame in frames.items():
        copied = frame.copy()
        copied.index = _normalize_index(copied.index)
        matches = copied.loc[copied.index == target]
        if len(matches) != 1:
            raise LiveDataUnavailable(f"exact target {target} is not available for feature/{alpha}")
        result[alpha] = matches
    return result


def load_feature_step_window(
    data_source: Any, task_time: str, feature_plan: FeaturePlan, *,
    wait_timeout_seconds: float | None = None, wait_poll_seconds: float = 1.0) -> FeatureWindow:

    resolved_task_time = task_time or datetime.now().strftime("%Y-%m-%d %H:00:00")
    requested_time = _parse_task_timestamp(resolved_task_time)
    deadline = None if wait_timeout_seconds is None else time.monotonic() + max(0.0, float(wait_timeout_seconds))
    step_plan = _feature_plan_with_seq_len(feature_plan, 1)
    last_error: Exception | None = None

    while True:
        date = _resolve_feature_date(data_source, resolved_task_time)
        try:
            direct_loader = getattr(data_source, "load_feature_step_window", None)
            direct_window = False
            if callable(direct_loader):
                feature_window = direct_loader(date, feature_plan, resolved_task_time)
                direct_window = True
            else:
                frames = _load_feature_step_frames(data_source, date, feature_plan.alphas, resolved_task_time)
                feature_window = build_feature_window(frames, step_plan, resolved_task_time)
        except LiveDataUnavailable as exc:
            last_error = exc
        else:
            if requested_time is None or pd.Timestamp(feature_window.selected_time) == pd.Timestamp(requested_time):
                if direct_window:
                    return feature_window
                return FeatureWindow(
                    values=feature_window.values,
                    columns=feature_window.columns,
                    index=feature_window.index,
                    selected_time=feature_window.selected_time,
                    requested_time=feature_window.requested_time,
                    task_time_floor_used=False,
                    feature_lengths=feature_window.feature_lengths,
                    source_date=date,
                )
            last_error = LiveDataUnavailable(
                f"exact target {requested_time} is not available; latest feature row is {feature_window.selected_time}"
            )

        if deadline is not None and time.monotonic() >= deadline:
            raise LiveDataUnavailable(f"exact target {requested_time} is not available before timeout") from last_error
        time.sleep(max(0.0, float(wait_poll_seconds)))


def combine_history_and_latest_feature_window(
    history_window: FeatureWindow, latest_window: FeatureWindow, feature_plan: FeaturePlan) -> FeatureWindow:

    if history_window.columns != latest_window.columns:
        raise ValueError("history/latest columns mismatch")
    if history_window.values.shape[:2] != latest_window.values.shape[:2]:
        raise ValueError("history/latest feature shape mismatch")
    if history_window.values.shape[-1] != feature_plan.seq_len - 1:
        raise ValueError(
            f"history window length {history_window.values.shape[-1]} != seq_len-1 {feature_plan.seq_len - 1}"
        )
    if latest_window.values.shape[-1] != 1:
        raise ValueError(f"latest window must contain exactly one row, got {latest_window.values.shape[-1]}")
    if pd.Timestamp(history_window.index[-1]) != _expected_history_end_for_target(latest_window.selected_time):
        raise ValueError("history window does not end at target-1h")

    values = np.concatenate([history_window.values, latest_window.values], axis=-1)
    index = history_window.index.append(latest_window.index)
    return FeatureWindow(
        values=values,
        columns=list(history_window.columns),
        index=index,
        selected_time=latest_window.selected_time,
        requested_time=latest_window.requested_time,
        task_time_floor_used=False,
        feature_lengths=latest_window.feature_lengths,
        source_date=latest_window.source_date or history_window.source_date,
    )


def dump_feature_frames(output_dir: str | Path, frames: dict[str, pd.DataFrame], metadata: dict[str, Any] | None = None) -> Path:
    out = Path(output_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_files = {}
    frame_info = {}
    for idx, (alpha, frame) in enumerate(frames.items()):
        filename = f"{idx:03d}_{quote(alpha, safe='')}.pkl"
        rel_path = Path("frames") / filename
        frame.to_pickle(out / rel_path)
        frame_files[alpha] = str(rel_path)
        frame_info[alpha] = {
            "shape": list(frame.shape),
            "columns": [str(col) for col in frame.columns],
            "columns_count": int(len(frame.columns)),
            "index_start": str(frame.index[0]) if len(frame.index) else None,
            "index_end": str(frame.index[-1]) if len(frame.index) else None,
            "index_count": int(len(frame.index)),
        }
    payload = {
        "dump_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alphas": list(frames.keys()),
        "frame_files": frame_files,
        "frame_info": frame_info,
        "metadata": metadata or {},
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return out


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize("Asia/Shanghai")
    else:
        dt_index = dt_index.tz_convert("Asia/Shanghai")
    return dt_index


def _build_aligned_feature_window_fast(
    frames: dict[str, pd.DataFrame], feature_plan: FeaturePlan, task_time: str | None) -> FeatureWindow | None:

    missing = [alpha for alpha in feature_plan.alphas if alpha not in frames]
    if missing:
        raise ValueError(f"missing SHM features: {missing}")
    reference_frame = frames[feature_plan.alphas[0]]
    if not isinstance(reference_frame, pd.DataFrame):
        raise ValueError(f"feature/{feature_plan.alphas[0]} is not a DataFrame: {type(reference_frame)}")
    if reference_frame.empty:
        raise ValueError(f"feature/{feature_plan.alphas[0]} is empty")

    reference_columns = [str(col) for col in reference_frame.columns]
    reference_index = _normalize_index(reference_frame.index)
    if not reference_index.is_monotonic_increasing or reference_index.has_duplicates:
        return None

    feature_lengths: dict[str, int] = {}
    for alpha in feature_plan.alphas:
        frame = frames[alpha]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"feature/{alpha} is not a DataFrame: {type(frame)}")
        if frame.empty:
            raise ValueError(f"feature/{alpha} is empty")
        columns = [str(col) for col in frame.columns]
        if columns != reference_columns:
            raise ValueError(f"columns mismatch for feature/{alpha}")
        feature_lengths[alpha] = len(frame)
        if alpha == feature_plan.alphas[0]:
            continue
        index = _normalize_index(frame.index)
        if len(index) != len(reference_index) or not index.equals(reference_index):
            return None

    requested_time = _parse_task_timestamp(task_time)
    if requested_time is None:
        selected_pos = len(reference_index) - 1
        selected_time = reference_index[selected_pos]
        task_time_floor_used = False
    else:
        selected_pos = int(reference_index.searchsorted(requested_time, side="right")) - 1
        if selected_pos < 0:
            raise ValueError(f"no feature row <= task_time {requested_time}")
        selected_time = reference_index[selected_pos]
        task_time_floor_used = selected_time != requested_time

    start_pos = selected_pos - feature_plan.seq_len + 1
    if start_pos < 0:
        raise ValueError(
            f"not enough aligned rows for seq_len={feature_plan.seq_len}: "
            f"selected_pos={selected_pos}, aligned_rows={len(reference_index)}"
        )
    window_index = reference_index[start_pos : selected_pos + 1]
    values = np.empty((len(reference_columns), len(feature_plan.alphas), feature_plan.seq_len), dtype=np.float32)
    row_slice = slice(start_pos, selected_pos + 1)
    for channel, alpha in enumerate(feature_plan.alphas):
        values[:, channel, :] = frames[alpha].iloc[row_slice].to_numpy(dtype=np.float32, copy=False).T

    return FeatureWindow(
        values=values, columns=reference_columns, index=window_index, selected_time=selected_time, requested_time=requested_time,
        task_time_floor_used=task_time_floor_used, feature_lengths=feature_lengths)


def build_feature_window(frames: dict[str, pd.DataFrame], feature_plan: FeaturePlan, task_time: str | None) -> FeatureWindow:
    fast_window = _build_aligned_feature_window_fast(frames, feature_plan, task_time)
    if fast_window is not None:
        return fast_window

    missing = [alpha for alpha in feature_plan.alphas if alpha not in frames]
    if missing:
        raise ValueError(f"missing SHM features: {missing}")

    reference_columns: list[str] | None = None
    normalized: dict[str, pd.DataFrame] = {}
    feature_lengths: dict[str, int] = {}
    common_index: pd.DatetimeIndex | None = None
    for alpha in feature_plan.alphas:
        frame = frames[alpha]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"feature/{alpha} is not a DataFrame: {type(frame)}")
        if frame.empty:
            raise ValueError(f"feature/{alpha} is empty")
        columns = [str(col) for col in frame.columns]
        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(f"columns mismatch for feature/{alpha}")

        copied = frame.copy()
        copied.columns = columns
        copied.index = _normalize_index(copied.index)
        copied = copied.sort_index()
        copied = copied[~copied.index.duplicated(keep="last")]
        normalized[alpha] = copied
        feature_lengths[alpha] = len(copied)
        common_index = copied.index if common_index is None else common_index.intersection(copied.index)

    assert common_index is not None
    common_index = pd.DatetimeIndex(common_index).sort_values()
    if common_index.empty:
        raise ValueError("features have no common aligned index")

    requested_time = _parse_task_timestamp(task_time)
    if requested_time is None:
        selected_time = common_index[-1]
        task_time_floor_used = False
    else:
        eligible = common_index[common_index <= requested_time]
        if eligible.empty:
            raise ValueError(f"no feature row <= task_time {requested_time}")
        selected_time = eligible[-1]
        task_time_floor_used = selected_time != requested_time

    selected_pos = common_index.get_loc(selected_time)
    if isinstance(selected_pos, slice) or not isinstance(selected_pos, (int, np.integer)):
        raise ValueError(f"selected_time is not unique in aligned index: {selected_time}")
    start_pos = int(selected_pos) - feature_plan.seq_len + 1
    if start_pos < 0:
        raise ValueError(
            f"not enough aligned rows for seq_len={feature_plan.seq_len}: "
            f"selected_pos={selected_pos}, aligned_rows={len(common_index)}"
        )
    window_index = common_index[start_pos : int(selected_pos) + 1]
    values = np.empty((len(reference_columns or []), len(feature_plan.alphas), feature_plan.seq_len), dtype=np.float32)
    for channel, alpha in enumerate(feature_plan.alphas):
        values[:, channel, :] = normalized[alpha].loc[window_index, reference_columns].to_numpy(dtype=np.float32).T

    return FeatureWindow(
        values=values,
        columns=reference_columns or [],
        index=window_index,
        selected_time=selected_time,
        requested_time=requested_time,
        task_time_floor_used=task_time_floor_used,
        feature_lengths=feature_lengths,
    )


def load_feature_window(
    data_source: Any, task_time: str | None, feature_plan: FeaturePlan, *,
    dump_dir: str | Path | None = None, dump_metadata: dict[str, Any] | None = None,
    exact_task_time: bool = False, feature_date_override: str | None = None, universe_date: str | None = None,
    wait_timeout_seconds: float | None = None, wait_poll_seconds: float = 1.0) -> FeatureWindow:

    resolved_task_time = task_time or datetime.now().strftime("%Y-%m-%d %H:00:00")
    requested_time = _parse_task_timestamp(resolved_task_time)
    deadline = None if wait_timeout_seconds is None else time.monotonic() + max(0.0, float(wait_timeout_seconds))
    last_error: Exception | None = None

    while True:
        date = (
            str(feature_date_override)
            if feature_date_override is not None
            else _resolve_feature_date(data_source, resolved_task_time)
        )
        universe_loader = getattr(data_source, "load_feature_frames_for_universe", None)

        if universe_date is not None and callable(universe_loader):
            frames = universe_loader(date, feature_plan.alphas, str(universe_date))
        else:
            frames = data_source.load_feature_frames(date, feature_plan.alphas)

        try:
            feature_window = build_feature_window(frames, feature_plan, resolved_task_time)
        except ValueError as exc:
            message = str(exc)
            retryable = any(
                fragment in message
                for fragment in (" is empty", "no feature row <= task_time", "not enough aligned rows", "features have no common aligned index")
            )
            if not exact_task_time or not retryable:
                raise
            last_error = exc
        else:
            if (not exact_task_time or requested_time is None or pd.Timestamp(feature_window.selected_time) == pd.Timestamp(requested_time)):
                feature_window = FeatureWindow(
                    values=feature_window.values, columns=feature_window.columns, index=feature_window.index,
                    selected_time=feature_window.selected_time, requested_time=feature_window.requested_time,
                    task_time_floor_used=feature_window.task_time_floor_used, feature_lengths=feature_window.feature_lengths, source_date=date)

                if dump_dir is not None:
                    metadata = {
                        "date": date,
                        "task_time": resolved_task_time,
                        "selected_time": str(feature_window.selected_time),
                        "source": getattr(data_source, "source_name", data_source.__class__.__name__),
                        **(dump_metadata or {}),
                    }
                    dump_feature_frames(dump_dir, frames, metadata=metadata)
                return feature_window
            last_error = LiveDataUnavailable(
                f"exact target {requested_time} is not available; latest feature row is {feature_window.selected_time}"
            )

        invalidate = getattr(data_source, "invalidate_feature_frame_cache", None)
        if callable(invalidate):
            invalidate()
        if deadline is not None and time.monotonic() >= deadline:
            raise LiveDataUnavailable(f"exact target {requested_time} is not available before timeout") from last_error
        time.sleep(max(0.0, float(wait_poll_seconds)))


def prepared_cache_status(cached: CachedCheckpointModel, feature_window: FeatureWindow) -> tuple[bool, str | None]:
    if len(feature_window.index) < 2:
        return False, "window_too_short"
    if cached.prepared_target_time is None:
        return False, "not_prepared"
    if cached.cache_consumed:
        return False, "cache_consumed"
    target = pd.Timestamp(feature_window.selected_time)
    if pd.Timestamp(cached.prepared_target_time) != target:
        return False, "target_mismatch"
    expected_history_end = pd.Timestamp(feature_window.index[-2])
    if cached.prepared_history_end_time is None or pd.Timestamp(cached.prepared_history_end_time) != expected_history_end:
        return False, "history_end_mismatch"
    if cached.prepared_columns != list(feature_window.columns):
        return False, "columns_mismatch"
    return True, None


def feature_window_to_frames(feature_window: FeatureWindow, feature_plan: FeaturePlan) -> dict[str, pd.DataFrame]:
    if feature_window.values.shape[1] != len(feature_plan.alphas):
        raise ValueError(f"feature window channel count {feature_window.values.shape[1]} != alpha count {len(feature_plan.alphas)}")

    return {
        alpha: pd.DataFrame(feature_window.values[:, channel, :].T, index=feature_window.index, columns=feature_window.columns)
        for channel, alpha in enumerate(feature_plan.alphas)
    }


def infer_prepared_cached_checkpoint_raw(
    cached: CachedCheckpointModel, feature_window: FeatureWindow, *, cache_hit: bool = True) -> tuple[np.ndarray, np.ndarray]:

    ok, reason = prepared_cache_status(cached, feature_window)
    if not ok:
        cached.last_cache_hit = False
        cached.last_rebuild_reason = reason
        raise ValueError(f"prepared cache is not usable: {reason}")
    norm_feature = _normalize_cached_feature(cached, feature_window, slice(-1, None))
    pred, prob = _forward_cached_model(cached, norm_feature)
    cached.cache_consumed = True
    cached.last_cache_hit = bool(cache_hit)
    if cache_hit:
        cached.last_rebuild_reason = None
    return pred, prob


def infer_cached_checkpoint_raw(cached: CachedCheckpointModel, feature_window: FeatureWindow) -> tuple[np.ndarray, np.ndarray]:
    cache_hit, reason = prepared_cache_status(cached, feature_window)
    if not cache_hit:
        prewarm_cached_checkpoint_for_target(cached, feature_window, feature_window.selected_time)
        cached.last_rebuild_reason = reason
    return infer_prepared_cached_checkpoint_raw(cached, feature_window, cache_hit=cache_hit)


def online_ewm_vola_scale(ret1m_window: np.ndarray, horizon: int) -> np.ndarray:
    if ret1m_window.ndim != 2:
        raise ValueError(f"ret1m_window must be 2-D [instrument, time], got {ret1m_window.shape}")
    horizon = max(10, int(horizon))
    alpha = 1 - np.exp(-np.log(2) / (horizon * 2))
    ret = ret1m_window.astype(np.float32, copy=True)
    ewm_mean = np.zeros(ret.shape[0], dtype=np.float32)
    ewm_var = np.zeros(ret.shape[0], dtype=np.float32)
    initialized = np.zeros(ret.shape[0], dtype=bool)
    for t in range(ret.shape[1]):
        curr = ret[:, t]
        curr_valid = np.isfinite(curr)
        curr = curr.copy()
        curr[~curr_valid] = 0.0
        first_valid = curr_valid & ~initialized
        ewm_mean[first_valid] = curr[first_valid]
        ewm_var[first_valid] = 0.0
        initialized[first_valid] = True
        update_valid = curr_valid & ~first_valid
        if not update_valid.any():
            continue
        new_mean = alpha * curr + (1 - alpha) * ewm_mean
        new_var = alpha * ((curr - new_mean) ** 2) + (1 - alpha) * ewm_var
        ewm_mean[update_valid] = new_mean[update_valid]
        ewm_var[update_valid] = new_var[update_valid]
    return np.log1p(ewm_var**0.5)


def _decode_model_pred_with_vola_scale(
    pred: np.ndarray, label_std: np.ndarray, valid: np.ndarray, vola_scale: np.ndarray | float) -> np.ndarray:
    decoded = label_std.reshape(1, -1)[:, 0] * pred * vola_scale
    decoded = np.expm1(np.abs(decoded)) * np.sign(decoded)
    decoded[~valid] = 0.0
    decoded[~np.isfinite(decoded)] = 0.0
    return decoded


def decode_model_pred(
    pred: np.ndarray, label_std: np.ndarray, valid: np.ndarray, *, ret1m_window: np.ndarray | None, horizon: int, use_vola: bool) -> tuple[np.ndarray, dict[str, Any]]:

    if use_vola:
        if ret1m_window is None:
            raise ValueError("ret1m_window is required when use_vola=True")
        vola_scale: np.ndarray | float = online_ewm_vola_scale(ret1m_window, horizon)
    else:
        vola_scale = 1.0
    decoded = _decode_model_pred_with_vola_scale(pred, label_std, valid, vola_scale)
    return decoded, {"vola_scale": vola_scale}


def _build_checkpoint_signal_result_from_decoded(
    cached: CachedCheckpointModel, feature_window: FeatureWindow, decoded: np.ndarray, prob: np.ndarray, valid: np.ndarray,
    vola_scale: np.ndarray | float) -> tuple[pd.Series, pd.Series, dict[str, Any]]:

    signal = pd.Series(decoded, index=feature_window.columns, name=cached.info.version)
    prob_signal = pd.Series(prob, index=feature_window.columns, name=cached.info.version)
    stats = {
        "version": cached.info.version,
        "pred_min": float(np.nanmin(decoded)),
        "pred_max": float(np.nanmax(decoded)),
        "pred_mean": float(np.nanmean(decoded)),
        "pred_std": float(np.nanstd(decoded)),
        "prob_min": float(np.nanmin(prob)),
        "prob_max": float(np.nanmax(prob)),
        "prob_mean": float(np.nanmean(prob)),
        "prob_std": float(np.nanstd(prob)),
        "valid_count": int(valid.sum()),
        "nan_count": int(np.isnan(decoded).sum()),
        "inf_count": int(np.isinf(decoded).sum()),
        "use_vola": cached.use_vola,
        "vola_horizon": int(cached.label_horizons[0]),
        "vola_scale_min": float(np.nanmin(vola_scale)),
        "vola_scale_max": float(np.nanmax(vola_scale)),
        "vola_scale_mean": float(np.nanmean(vola_scale)),
        "vola_scale_std": float(np.nanstd(vola_scale)),
        "cache_hit": bool(cached.last_cache_hit),
        "rebuild_reason": cached.last_rebuild_reason,
        "prepared_steps": int(cached.last_prepared_steps),
    }
    return signal, prob_signal, stats


def _build_checkpoint_signal_result(
    cached: CachedCheckpointModel, feature_window: FeatureWindow, pred: np.ndarray, prob: np.ndarray) -> tuple[pd.Series, pd.Series, dict[str, Any]]:

    ret1m_window = feature_window.values[:, 0, :]
    valid = np.isfinite(ret1m_window[:, -1])
    decoded, decode_stats = decode_model_pred(
        pred, cached.label_std, valid, ret1m_window=ret1m_window, horizon=cached.label_horizons[0], use_vola=cached.use_vola)

    return _build_checkpoint_signal_result_from_decoded(cached, feature_window, decoded, prob, valid, decode_stats["vola_scale"])


def infer_cached_checkpoint_signal(cached: CachedCheckpointModel, feature_window: FeatureWindow) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    pred, prob = infer_cached_checkpoint_raw(cached, feature_window)
    return _build_checkpoint_signal_result(cached, feature_window, pred, prob)


def infer_signal_from_feature_window(cached_models: list[CachedCheckpointModel], feature_window: FeatureWindow) -> InferenceResult:

    if not all(isinstance(cached, CachedCheckpointModel) for cached in cached_models):
        outputs = _parallel_map_cached_models(cached_models, feature_window, infer_cached_checkpoint_signal)
        version_signals = [signal for signal, _prob, _stats in outputs]
        version_stats = [stats for _signal, _prob, stats in outputs]
    else:
        raw_outputs = _parallel_map_cached_models(cached_models, feature_window, infer_cached_checkpoint_raw)
        ret1m_window = feature_window.values[:, 0, :]
        valid = np.isfinite(ret1m_window[:, -1])
        vola_scales: dict[tuple[bool, int], np.ndarray | float] = {}
        version_signals = []
        version_stats = []
        for cached, (pred, prob) in zip(cached_models, raw_outputs, strict=True):
            key = (cached.use_vola, int(cached.label_horizons[0]))
            if key not in vola_scales:
                if cached.use_vola:
                    vola_scales[key] = online_ewm_vola_scale(ret1m_window, cached.label_horizons[0])
                else:
                    vola_scales[key] = 1.0
            vola_scale = vola_scales[key]
            decoded = _decode_model_pred_with_vola_scale(pred, cached.label_std, valid, vola_scale)
            signal, _prob_signal, stats = _build_checkpoint_signal_result_from_decoded(
                cached, feature_window, decoded, prob, valid, vola_scale)
            version_signals.append(signal)
            version_stats.append(stats)
    fused = sum(version_signals) / len(version_signals)
    frame = pd.DataFrame([fused.values], index=[feature_window.selected_time], columns=feature_window.columns)
    return InferenceResult(signal=frame, selected_time=feature_window.selected_time, feature_window=feature_window, version_stats=version_stats)


def calculate_signal(runtime: SignalOnlyRuntime, feature_window: FeatureWindow) -> InferenceResult:
    return infer_signal_from_feature_window(runtime.cached_models, feature_window)


def run_once(args: argparse.Namespace, loop_state: LoopState | None = None, runtime: SignalOnlyRuntime | None = None) -> Path | None:
    runtime = runtime or prepare_signal_runtime(args)
    infos = runtime.infos
    feature_plan = runtime.feature_plan
    writer = runtime.writer
    task_time = args.task_time or datetime.now().strftime("%Y-%m-%d %H:00:00")
    date = pd.Timestamp(task_time).strftime("%Y-%m-%d")

    if args.fake_signal:
        universe = _load_universe_for_fake(args.fake_universe_file)
        signal = _fake_signal(universe, task_time)
        is_duplicate = loop_state is not None and not loop_state.should_write(signal.index[0])
        if is_duplicate and not args.debug_replay_duplicates:
            print(f"HEARTBEAT duplicate selected_time={signal.index[0]}; no audit write", flush=True)
            return None
        if is_duplicate:
            print(
                f"HEARTBEAT duplicate selected_time={signal.index[0]}; debug replay enabled; audit write will run",
                flush=True,
            )
        write_result = writer.write(
            date=date, universe_name=args.universe, signal_key=args.signal_key, signal=signal, versions=[info.version for info in infos],
            feature_plan=feature_plan, task_time=task_time, extra={"fake_signal": True, "reason": "explicit --fake-signal mode"}
        )
        if loop_state is not None:
            loop_state.mark_written(signal.index[0])
        return write_result

    data_source = runtime.data_source or make_data_source(args)
    requested_selected_time = _parse_task_timestamp(task_time)
    if (loop_state is not None and requested_selected_time is not None
        and loop_state.last_written_selected_time == requested_selected_time and not args.debug_replay_duplicates):
        prepare_next_target_cache(
            runtime, data_source, requested_selected_time, wait_timeout_seconds=SPECULATIVE_PREWARM_TIMEOUT_SECONDS,
            wait_poll_seconds=getattr(args, "wait_poll_seconds", 1.0)
        )
        print(f"HEARTBEAT duplicate selected_time={requested_selected_time}; no audit write; inference skipped", flush=True)
        return None

    load_start = time.perf_counter()
    prewarm_ms = 0.0
    latest_step_ms = 0.0
    if _supports_prepared_runtime(runtime.cached_models) and requested_selected_time is not None:
        target_feature_date = _resolve_feature_date(data_source, task_time)
        history_window = _prepared_runtime_history_window(runtime.cached_models, requested_selected_time)
        if history_window is None:
            prewarm_start = time.perf_counter()
            history_window = load_prewarm_feature_window(
                data_source, task_time, feature_plan, target_feature_date=target_feature_date,
                wait_timeout_seconds=getattr(args, "wait_timeout_seconds", None), wait_poll_seconds=getattr(args, "wait_poll_seconds", 1.0),
            )
            _parallel_map_cached_models(
                runtime.cached_models, history_window,
                lambda cached, window: prewarm_cached_checkpoint_for_target(cached, window, requested_selected_time)
            )
            step_prepare = getattr(data_source, "prepare_feature_step_reader", None)
            if callable(step_prepare):
                step_prepare(target_feature_date, feature_plan.alphas)
            prewarm_ms = (time.perf_counter() - prewarm_start) * 1e3
        latest_start = time.perf_counter()
        latest_window = load_feature_step_window(
            data_source, task_time, feature_plan, wait_timeout_seconds=getattr(args, "wait_timeout_seconds", None),
            wait_poll_seconds=getattr(args, "wait_poll_seconds", 1.0)
        )
        latest_step_ms = (time.perf_counter() - latest_start) * 1e3
        feature_window = combine_history_and_latest_feature_window(history_window, latest_window, feature_plan)
        if getattr(args, "dump_feature_frames", None) is not None:
            dump_feature_frames(
                args.dump_feature_frames, feature_window_to_frames(feature_window, feature_plan),
                metadata={
                    "date": feature_window.source_date,
                    "task_time": task_time,
                    "selected_time": str(feature_window.selected_time),
                    "source": getattr(data_source, "source_name", data_source.__class__.__name__),
                    "signal_key": args.signal_key,
                    "universe": args.universe,
                }
            )
    else:
        feature_window = load_feature_window(
            data_source, task_time, feature_plan, dump_dir=getattr(args, "dump_feature_frames", None),
            dump_metadata={
                "signal_key": args.signal_key,
                "universe": args.universe,
            },
            exact_task_time=True,
            wait_timeout_seconds=getattr(args, "wait_timeout_seconds", None),
            wait_poll_seconds=getattr(args, "wait_poll_seconds", 1.0),
        )

    load_ms = (time.perf_counter() - load_start) * 1e3

    is_duplicate = loop_state is not None and not loop_state.should_write(feature_window.selected_time)
    if is_duplicate and not args.debug_replay_duplicates:
        print(f"HEARTBEAT duplicate selected_time={feature_window.selected_time}; no audit write; inference skipped", flush=True)
        return None
    if is_duplicate:
        print(f"HEARTBEAT duplicate selected_time={feature_window.selected_time}; debug replay enabled; inference will run", flush=True)

    infer_start = time.perf_counter()
    result = calculate_signal(runtime, feature_window)
    infer_ms = (time.perf_counter() - infer_start) * 1e3
    source_date = result.feature_window.source_date or result.selected_time.strftime("%Y-%m-%d")
    cache_hits = [bool(stats.get("cache_hit")) for stats in result.version_stats]
    rebuild_reasons = [stats.get("rebuild_reason") for stats in result.version_stats]

    write_result = writer.write(
        date=source_date,
        universe_name=args.universe,
        signal_key=args.signal_key,
        signal=result.signal,
        versions=[info.version for info in infos],
        feature_plan=feature_plan,
        task_time=task_time,
        extra={
            "fake_signal": False,
            "source": getattr(data_source, "source_name", data_source.__class__.__name__),
            "source_date": source_date,
            "shm_path": f"/data/f2/crypto/shm/{source_date}/1h" if getattr(data_source, "source_name", "") == "shm" else None,
            "dump_dir": str(getattr(args, "dump_dir", "")) if getattr(args, "source", "shm") == "dump" else None,
            "dump_feature_frames": str(getattr(args, "dump_feature_frames", "")) if getattr(args, "dump_feature_frames", None) else None,
            "requested_time": str(result.feature_window.requested_time) if result.feature_window.requested_time is not None else None,
            "selected_time": str(result.selected_time),
            "task_time_floor_used": result.feature_window.task_time_floor_used,
            "window_start": str(result.feature_window.index[0]),
            "window_end": str(result.feature_window.index[-1]),
            "window_rows": len(result.feature_window.index),
            "feature_count": len(feature_plan.alphas),
            "feature_lengths": result.feature_window.feature_lengths,
            "model_input_shape": list(result.feature_window.values.shape),
            "fusion": "8-version arithmetic mean of decoded pred channel 0; prob is computed per version but not fused into signal",
            "version_stats": result.version_stats,
            "cache_hit": bool(cache_hits) and all(cache_hits),
            "rebuild_reasons": rebuild_reasons,
            "timings_ms": {
                "load_feature_window": load_ms,
                "prewarm": prewarm_ms,
                "load_latest_step": latest_step_ms,
                "calculate_signal": infer_ms,
            },
        },
    )

    if loop_state is not None:
        loop_state.mark_written(result.selected_time)
        loop_state.last_feature_window = result.feature_window

    return write_result


def loop_sleep_seconds(
    args: argparse.Namespace, now: pd.Timestamp | None = None, *, retry_after_failure: bool = False,
    last_written_selected_time: pd.Timestamp | None = None) -> float:

    if retry_after_failure or getattr(args, "task_time", None) is not None:
        return max(1.0, float(args.sleep_seconds))
    current = now if now is not None else pd.Timestamp.now(tz="Asia/Shanghai")
    current = pd.Timestamp(current)
    if current.tzinfo is None:
        current = current.tz_localize("Asia/Shanghai")
    else:
        current = current.tz_convert("Asia/Shanghai")
    if last_written_selected_time is not None:
        last_written = pd.Timestamp(last_written_selected_time)
        if last_written.tzinfo is None:
            last_written = last_written.tz_localize("Asia/Shanghai")
        else:
            last_written = last_written.tz_convert("Asia/Shanghai")
        if last_written < current.floor("h"):
            return 1.0
    next_hour = current.floor("h") + pd.Timedelta(hours=1)
    return max(0.0, float((next_hour - current).total_seconds()))


def main():
    args = parse_args()

    last_selected_time = initial_loop_selected_time(args)
    if last_selected_time is not None:
        print(f"HEARTBEAT seeded last_written_selected_time={last_selected_time}", flush=True)

    runtime = prepare_signal_runtime(args)
    loop_state = LoopState(last_selected_time) if args.loop else None

    while True:
        retry_after_failure = False
        try:
            path = run_once(args, loop_state=loop_state, runtime=runtime)
            if path is not None:
                print(path, flush=True)
                if loop_state is not None and loop_state.last_written_selected_time is not None:
                    try:
                        prepare_next_target_cache(
                            runtime, runtime.data_source or make_data_source(args),
                            loop_state.last_written_selected_time,
                            current_feature_window=loop_state.last_feature_window,
                            wait_timeout_seconds=SPECULATIVE_PREWARM_TIMEOUT_SECONDS,
                            wait_poll_seconds=getattr(args, "wait_poll_seconds", 1.0),
                        )
                        next_target = loop_state.last_written_selected_time + pd.Timedelta(hours=1)
                        print(f"HEARTBEAT prepared_next_target={next_target}", flush=True)
                    except LiveDataUnavailable as exc:
                        print(f"PREWARM_NEXT_FAILED: {exc}", flush=True)
                        retry_after_failure = True
        except LiveDataUnavailable as exc:
            print(f"LIVE_DATA_UNAVAILABLE: {exc}", flush=True)
            retry_after_failure = True
            if args.once:
                raise SystemExit(2)
        if args.once:
            return
        time.sleep(
            loop_sleep_seconds(
                args, retry_after_failure=retry_after_failure,
                last_written_selected_time=(loop_state.last_written_selected_time if loop_state is not None else None))
        )

if __name__ == "__main__":
    main()
