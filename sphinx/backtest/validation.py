import pandas as pd
import numpy as np
from typing import Any

# from sphinx.util.exchange_api import read_m

Frame = pd.DataFrame


def require_number(value: Any, label: str) -> float:
    """配置里的数字必须能转成有限浮点数。"""

    if isinstance(value, bool):
        raise SystemExit(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be a finite number") from exc
    if not np.isfinite(number):
        raise SystemExit(f"{label} must be a finite number")
    return number


def require_positive_number(value: Any, label: str) -> float:
    """配置里的正数必须是有限且大于 0。"""

    number = require_number(value, label)
    if number <= 0.0:
        raise SystemExit(f"{label} must be positive")
    return number

def require_mapping(value: Any, label: str) -> dict[str, Any]:
    """配置里的分组必须是对象，不能是空值或其他类型。"""

    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def require_config_value(mapping: dict[str, Any], key: str, expected: Any, label: str) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise SystemExit(f"{label}.{key} must be {expected!r}, got {actual!r}")


def require_nonempty_string(value: Any, label: str) -> str:
    """配置里的字符串必须非空。"""

    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label} must be a non-empty string")
    return value


def require_bool(value: Any, label: str) -> bool:
    """配置里的布尔开关必须是真的 bool，避免字符串误写后静默生效。"""

    if not isinstance(value, bool):
        raise SystemExit(f"{label} must be a boolean")
    return value


def require_nonnegative_number(value: Any, label: str) -> float:
    """配置或数据里的非负数必须是有限且大于等于 0。"""

    number = require_number(value, label)
    if number < 0.0:
        raise SystemExit(f"{label} must be non-negative")
    return number

def invalid_number_mask(value: Frame) -> Frame:
    """返回 NaN、正无穷、负无穷或非数字位置。"""

    try:
        numeric = value.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame must contain numeric values") from exc
    return pd.DataFrame(
        ~np.isfinite(numeric.to_numpy()),
        index=value.index,
        columns=value.columns,
    )

    
def first_bad_location(mask: Frame) -> tuple[Any, Any]:
    """返回布尔矩阵里第一个 True 的坐标。"""
    return mask.stack(future_stack=True)[lambda data: data].index[0]


def require_no_nan(value: Frame, label: str) -> None:
    """最终要参与汇总的矩阵必须是有限数。

    pandas 的 sum 默认会跳过 NaN。回测里这很危险：一个坏成本可能被静默当成 0。
    所以在成本、换手和 pnl 出口处做一次硬检查。
    """

    missing = value.isna()
    if missing.any().any():
        timestamp, instrument = first_bad_location(missing)
        raise ValueError(f"{label} has NaN at {timestamp} {instrument}")
    invalid = invalid_number_mask(value)
    if invalid.any().any():
        timestamp, instrument = first_bad_location(invalid)
        raise ValueError(f"{label} has non-finite value at {timestamp} {instrument}")


def require_quantity_frame(value: Any, label: str) -> None:
    """数量仓位文件必须是非空、无重复、全有限的宽表。"""

    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{label} must be a DataFrame")
    if len(value.index) == 0:
        raise ValueError(f"{label} must have at least one row")
    if len(value.columns) == 0:
        raise ValueError(f"{label} must have at least one instrument")
    if value.index.has_duplicates:
        raise ValueError(f"{label} index has duplicate timestamps")
    if value.columns.has_duplicates:
        raise ValueError(f"{label} columns have duplicate instruments")
    require_no_nan(value, label)


def require_index_equal(actual: pd.Index, expected: pd.Index, label: str) -> None:
    """两个轴必须标签和顺序完全一致，忽略 Index.name/freq 这类元信息。"""

    actual = pd.Index(actual)
    expected = pd.Index(expected)
    missing = expected.difference(actual).tolist()
    extra = actual.difference(expected).tolist()
    if not missing and not extra and len(actual) == len(expected):
        if all(actual_value == expected_value for actual_value, expected_value in zip(actual, expected)):
            return
    detail = "same labels but different order"
    if missing or extra:
        detail = f"missing={missing} extra={extra}"
    raise ValueError(f"{label}: {detail}")


def require_quantity_frame(value: Any, label: str) -> None:
    """数量仓位文件必须是非空、无重复、全有限的宽表。"""

    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{label} must be a DataFrame")
    if len(value.index) == 0:
        raise ValueError(f"{label} must have at least one row")
    if len(value.columns) == 0:
        raise ValueError(f"{label} must have at least one instrument")
    if value.index.has_duplicates:
        raise ValueError(f"{label} index has duplicate timestamps")
    if value.columns.has_duplicates:
        raise ValueError(f"{label} columns have duplicate instruments")
    require_no_nan(value, label)


def require_data(value: Frame, used: Frame, label: str) -> None:
    """有交易或持仓时，相关数据必须是有限数。"""

    invalid = invalid_number_mask(value) & used
    if invalid.any().any():
        timestamp, instrument = first_bad_location(invalid)
        raise ValueError(f"{label} is missing or non-finite at {timestamp} {instrument}")


def require_positive_data(value: Frame, used: Frame, label: str) -> None:
    """有交易或持仓时，价格必须是有限正数。"""

    require_data(value, used, label)
    bad = (value <= 0.0) & used
    if bad.any().any():
        timestamp, instrument = first_bad_location(bad)
        raise ValueError(f"{label} must be positive at {timestamp} {instrument}")


def zero_nan_for_inactive_position(value: Frame, exposure: Frame, label: str) -> Frame:
    """只允许“没有持仓/没有交易”导致的坏值被当成 0。

    例子：
    - 新合约第一天没有上一日 close，但上一日数量也是 0，那么价格变化 NaN 不应产生 PnL。
    - 如果上一日数量非 0 却没有 close，这是真数据缺失，必须报错。
    """

    inactive = exposure.fillna(0.0).abs() == 0.0
    invalid = invalid_number_mask(value)
    result = value.mask(invalid & inactive, 0.0)
    remaining_invalid = invalid_number_mask(result)
    if remaining_invalid.any().any():
        timestamp, instrument = first_bad_location(remaining_invalid)
        raise ValueError(f"{label} has non-finite value for active exposure at {timestamp} {instrument}")
    return result


def invalid_number_mask(value: Frame) -> Frame:
    """返回 NaN、正无穷、负无穷或非数字位置。"""

    try:
        numeric = value.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame must contain numeric values") from exc
    return pd.DataFrame(
        ~np.isfinite(numeric.to_numpy()),
        index=value.index,
        columns=value.columns,
    )


def first_bad_location(mask: Frame) -> tuple[Any, Any]:
    """返回布尔矩阵里第一个 True 的坐标。"""
    return mask.stack(future_stack=True)[lambda data: data].index[0]


def require_nonnegative_data(value: Frame, used: Frame, label: str) -> None:
    """有交易或持仓时，成交额必须是有限非负数。"""

    require_data(value, used, label)
    bad = (value < 0.0) & used
    if bad.any().any():
        timestamp, instrument = first_bad_location(bad)
        raise ValueError(f"{label} must be non-negative at {timestamp} {instrument}")


def require_finite_nonnegative_scalar(value: Any, label: str) -> float:
    """数据里的手续费等标量必须是有限非负数。"""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number
