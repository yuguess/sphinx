import os
import sys
import json5
from pathlib import Path
from typing import Any, Mapping

# REPO_ROOT = Path(__file__).resolve().parent
# RUNTIME_CONFIG_ENV = "RUNTIME_CONFIG"
# RUNTIME_ENV_KEYS = ["EXCHANGE", "FREQ", "MARKET", "DATA_ROOT", "DEPLOY_MODE", "USE_MP_CACHE", "ENV_FILE"]
CF_EXCHANGE = "CF"
OKX_EXCHANGE = "okx"
FREQ_5MIN = "5min"
FREQ_1S = "1s"
FREQ_1H = "1h"
SUPPORTED_FREQS = {FREQ_5MIN, FREQ_1S, FREQ_1H}
CF_MARKET = "futures"
OKX_MARKET = "crypto"


def get_env_exchange() -> str:
    return os.environ.get("EXCHANGE", "")


def get_env_freq() -> str:
    return os.environ.get("FREQ", "")


def require_runtime_value(runtime: Mapping[str, Any], key: str, expected: Any, label: str = "runtime", error_type: type[BaseException] = ValueError) -> None:
    actual = runtime.get(key)
    if actual != expected:
        raise error_type(f"{label}.{key} must be {expected!r}, got {actual!r}")


def require_runtime_config(runtime: Mapping[str, Any], label: str = "runtime", error_type: type[BaseException] = ValueError) -> None:
    exchange = runtime.get("EXCHANGE")
    if exchange == OKX_EXCHANGE:
        freq = runtime.get("FREQ")
        if freq not in SUPPORTED_FREQS:
            raise error_type(f"{label}.FREQ must be one of {sorted(SUPPORTED_FREQS)!r}, got {freq!r}")
        require_runtime_value(runtime, "MARKET", OKX_MARKET, label, error_type)
    elif exchange == CF_EXCHANGE:
        require_runtime_value(runtime, "FREQ", FREQ_5MIN, label, error_type)
        require_runtime_value(runtime, "MARKET", CF_MARKET, label, error_type)
    else:
        raise error_type(f"{label}.EXCHANGE must be {CF_EXCHANGE!r} or {OKX_EXCHANGE!r}, got {exchange!r}")


#### deprecated below ####


RUNTIME_CONFIG_ENV = "RUNTIME_CONFIG"
RUNTIME_ENV_KEYS = ["EXCHANGE", "FREQ", "MARKET", "DATA_ROOT", "DEPLOY_MODE", "USE_MP_CACHE", "ENV_FILE"]


def runtime_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    elif path is None:
        env_path = os.environ.get(RUNTIME_CONFIG_ENV)
        if env_path:
            return Path(env_path)
        raise ValueError(
            f"{RUNTIME_CONFIG_ENV} must be set explicitly, for example "
            "runtime.cf_5min.json5 or runtime.okx_5min.json5"
        )
    else:
        raise ValueError(f"unexpected runtime config path: {path}")


def load_runtime_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = runtime_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json5.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"runtime config root must be an object: {config_path}")
    runtime = config.get("runtime", config)
    if not isinstance(runtime, dict):
        raise ValueError(f"runtime config must be an object: {config_path}")
    return dict(runtime)


def apply_runtime_config(runtime: Mapping[str, Any] | None = None, *, path: str | Path | None = None, override: bool = False) -> dict[str, Any]:

    if runtime is not None:
        runtime_dict = dict(runtime)
    elif runtime is None:
        runtime_dict = load_runtime_config(path)
    else:
        raise ValueError("unexpected runtime state")
    pythonpath = runtime_dict.get("PYTHONPATH", [])
    if isinstance(pythonpath, str):
        pythonpath = pythonpath.split(os.pathsep)
    if not isinstance(pythonpath, list):
        raise ValueError("runtime.PYTHONPATH must be a string or list when provided")

    missing = [key for key in RUNTIME_ENV_KEYS if key not in runtime_dict]
    if missing:
        raise ValueError(f"runtime missing required keys: {missing}")

    pythonpath_items = [str(item) for item in pythonpath if str(item)]
    for item in reversed(pythonpath_items):
        while item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)
    for key in RUNTIME_ENV_KEYS:
        if override or key not in os.environ:
            os.environ[key] = str(runtime_dict[key])
    return runtime_dict


def require_supported_runtime(runtime: Mapping[str, Any], label: str = "runtime", error_type: type[BaseException] = ValueError) -> None:
    exchange = runtime.get("EXCHANGE")
    if exchange == OKX_EXCHANGE:
        freq = runtime.get("FREQ")
        if freq not in SUPPORTED_FREQS:
            raise error_type(f"{label}.FREQ must be one of {sorted(SUPPORTED_FREQS)!r}, got {freq!r}")
        require_runtime_value(runtime, "MARKET", OKX_MARKET, label, error_type)
    elif exchange == CF_EXCHANGE:
        require_runtime_value(runtime, "FREQ", FREQ_5MIN, label, error_type)
        require_runtime_value(runtime, "MARKET", CF_MARKET, label, error_type)
    else:
        raise error_type(f"{label}.EXCHANGE must be {CF_EXCHANGE!r} or {OKX_EXCHANGE!r}, got {exchange!r}")
