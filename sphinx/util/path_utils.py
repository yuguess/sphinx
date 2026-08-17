import os
from pathlib import Path

from .runtime_config import require_runtime_config, get_env_exchange, get_env_freq

# STRATEGY_DL_ROOT = Path(__file__).resolve().parent
STRATEGY_DL_ROOT = os.environ["STRATEGY_DL_ROOT"]
STRATEGY_DL_DATA_ROOT = os.environ["STRATEGY_DL_DATA_ROOT"]


def require_supported_runtime() -> None:
    require_runtime_config(
        {
            "EXCHANGE": get_env_exchange(),
            "FREQ": get_env_freq(),
            "MARKET": os.environ.get("MARKET", ""),
        }
    )

def data_namespace_parts() -> tuple[str, str]:
    # require_supported_runtime()
    return get_env_exchange(), get_env_freq()


def data_namespace() -> str:
    return "/".join(data_namespace_parts())


def config_dir() -> Path:
    exchange, freq = data_namespace_parts()
    return Path(STRATEGY_DL_ROOT) / "config" / exchange / freq


def deep_data_root() -> Path:
    # return Path(os.environ.get("STRATEGY_DL_DATA_ROOT", DEFAULT_DEEP_DATA_ROOT)) / "deep" / data_namespace()
    return Path(STRATEGY_DL_DATA_ROOT) / "deep" / data_namespace()
