import pandas as pd
import numpy as np


from sphinx.util.exchange_api import read_holding, read_universe, get_index
from .validation import Frame, first_bad_location, require_quantity_frame
from .io import read_close, read_valid_frame, require_ep_holding_axes_match


def read_ep_holding_pct(date: str, ep_holding_univ_name: str, ep_holding_key: str) -> Frame:
    ep_holding = read_holding(date, ep_holding_univ_name, ep_holding_key)
    require_quantity_frame(ep_holding, f"pct_ep_holding {date}")
    universe = read_universe(date, ep_holding_univ_name)
    expected_index = get_index(date)
    require_ep_holding_axes_match(ep_holding, universe, expected_index, date, ep_holding_univ_name)
    return ep_holding

def pct_holding_to_quantity(date: str, pct_holding: Frame, nav: float, previous_qty: pd.Series | None = None) -> Frame:
    close = read_close(date, pct_holding.columns).reindex(index=pct_holding.index, columns=pct_holding.columns)
    active = pct_holding.abs() > 0.0
    bad_active_price = active & (~np.isfinite(close) | close.le(0.0))
    if bad_active_price.any().any():
        timestamp, instrument = first_bad_location(bad_active_price)
        raise ValueError(f"bad close for active pct_ep_holding at {date} {timestamp} {instrument}")
    quantity = pct_holding * nav / close
    quantity = quantity.where(active, 0.0)
    valid = read_valid_frame(date, pct_holding.columns).reindex(
        index=pct_holding.index,
        columns=pct_holding.columns,
    ).eq(1)
    previous = pd.Series(0.0, index=pct_holding.columns)
    if previous_qty is not None:
        previous = previous_qty.reindex(pct_holding.columns, fill_value=0.0).astype(float)
    elif previous_qty is None:
        pass
    else:
        raise ValueError("unexpected previous_qty state")
    rows = []
    for timestamp in quantity.index:
        row = quantity.loc[timestamp].copy()
        valid_row = valid.loc[timestamp].astype(bool)
        row = row.where(valid_row, previous)
        rows.append(row)
        previous = row
    quantity = pd.DataFrame(rows, index=pct_holding.index, columns=pct_holding.columns)
    require_quantity_frame(quantity, f"converted_ep_holding {date}")
    return quantity


def read_quantity_frames_from_pct(dates: list[str], ep_holding_univ_name: str, ep_holding_key: str, nav: float) -> dict[str, Frame]:
    result = {}
    previous_qty = None
    for date in dates:
        quantity = pct_holding_to_quantity(
            date, read_ep_holding_pct(date, ep_holding_univ_name, ep_holding_key), nav, previous_qty=previous_qty)
        result[date] = quantity
        previous_qty = quantity.iloc[-1]
    return result
