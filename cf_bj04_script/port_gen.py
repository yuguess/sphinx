import os
import argparse
import numpy as np
import pandas as pd
import json5
import mosek
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache

from sphinx.util.runtime_config import get_env_exchange
from sphinx.util.exchange_api import read_signal, write_holding
from sphinx.util.cf_exchange_api import read_basedata, read_feature_frame, read_valid_frame, read_universe, get_index, get_dates


EPS = 1e-7

SUPPORTED_OUTPUT_HOLDING_MODES = {"qty", "pct"}
EXCHANGE = get_env_exchange()

os.environ["MOSEKLM_LICENSE_FILE"] = "mosek.lic"
moseck_env = mosek.Env()

def normalize_output_holding_mode(value):
    if value not in SUPPORTED_OUTPUT_HOLDING_MODES:
        raise ValueError(f"output_holding_mode must be one of {sorted(SUPPORTED_OUTPUT_HOLDING_MODES)}, got {value!r}")
    return value


def require_optional_bool(mapping, key, label, default=False):
    value = mapping.get(key, default)
    if value is True:
        return True
    elif value is False:
        return False
    else:
        raise ValueError(f"{label}.{key} must be a bool")


def line_series(value, index):
    if isinstance(value, pd.Series):
        return value.reindex(index).astype(float)
    return pd.Series(value, index=index, dtype=float)


def pct_line_to_qty(pct_line, close, nav, valid=None, previous_qty_line=None):
    close = line_series(close, pct_line.index)
    if valid is None:
        active = pct_line.abs() > 0.0
        bad_active_price = active & (~np.isfinite(close) | close.le(0.0))
        if bad_active_price.any():
            inst = bad_active_price.index[bad_active_price.to_numpy().argmax()]
            raise ValueError(f"bad close for active qty output {inst}")
        qty = pct_line * nav / close
        qty = qty.where(active, 0.0)
        if not np.isfinite(qty.to_numpy()).all():
            raise ValueError("qty output has non-finite value")
        return qty

    valid = pd.Series(valid, index=pct_line.index, dtype=bool)
    if previous_qty_line is None:
        qty = pd.Series(0.0, index=pct_line.index, dtype=float)
    else:
        qty = previous_qty_line.reindex(pct_line.index).fillna(0.0)
    active_valid = valid & pct_line.abs().gt(0.0)
    bad_active_price = active_valid & (~np.isfinite(close) | close.le(0.0))
    if bad_active_price.any():
        inst = bad_active_price.index[bad_active_price.to_numpy().argmax()]
        raise ValueError(f"bad close for valid qty output {inst}")
    qty.loc[valid] = 0.0
    qty.loc[active_valid] = pct_line.loc[active_valid] * nav / close.loc[active_valid]
    return qty


def drift_pct_lines_by_close(pct_lines, close, previous_close, valid):
    """按 close 的价格变化漂移 pct 状态，并让 invalid 标的保持原 pct。

    pct 是按账户净值计的仓位比例。同一份 qty 在价格变化后，对应的 pct
    应该按 close / previous_close 自然变化。invalid 标的不参与当 bar 的优化，
    因此这里不能用当 bar 的坏 close 计算，也不能更新 previous_close。
    """
    # 所有 line 必须对齐到优化器当前使用的 instrument 顺序。后面的 mask、
    # close 和 pct 运算都依赖相同 index，避免按位置错配。
    index = pct_lines[0].index
    close = line_series(close, index)
    previous_close = line_series(previous_close, index)
    valid = pd.Series(valid, index=index, dtype=bool)

    # valid 标的是本 bar 要进入优化器的标的。valid 上 close 必须是可用正数；
    # 如果这里出现 NaN/inf/<=0，说明输入数据本身不满足优化前提，直接 fail-fast。
    # invalid 上允许 close 是坏值，因为它们不参与本 bar 优化，也不会用 close 漂移。
    bad_valid_close = valid & (~np.isfinite(close) | close.le(0.0))
    if bad_valid_close.any():
        inst = bad_valid_close.index[bad_valid_close.to_numpy().argmax()]
        raise ValueError(f"bad close for valid pct drift {inst}")

    # 只有当前存在仓位的 pct line 才需要 previous_close 来计算自然漂移。
    # 空仓标的即使没有 previous_close，也不会产生价格漂移，后面 ratio 会保持 1。
    active = sum(line.abs() for line in pct_lines).gt(0.0)
    good_previous_close = np.isfinite(previous_close) & previous_close.gt(0.0)

    # valid 且有仓位时，previous_close 缺失会让 pct 漂移无法定义。
    # 这通常表示上一轮状态没有正确记录，或者数据切换处存在未处理的边界。
    missing_active_previous = valid & active & ~good_previous_close
    if missing_active_previous.any():
        inst = missing_active_previous.index[missing_active_previous.to_numpy().argmax()]
        raise ValueError(f"bad previous close for active pct drift {inst}")

    # 只有 valid 且 previous_close 可用的标的才应用 close / previous_close。
    # 其他位置 ratio 置为 1：invalid 保持原 pct，valid 空仓且没有 previous_close
    # 仍然保持 0，不引入额外 NaN。
    drift_mask = valid & good_previous_close
    drift_ratio = (close / previous_close).where(drift_mask, 1.0)
    drifted = [line * drift_ratio for line in pct_lines]

    # previous_close 是下一 bar 漂移的基准价，只在本 bar close 合法且参与优化时更新。
    # invalid 标的继续沿用旧 previous_close，避免坏 close 污染后续状态。
    return drifted, previous_close.where(~valid, close)


def current_gross_exposure_bound(pct_lines):
    gross = float(sum(line.abs().sum() for line in pct_lines))
    if not np.isfinite(gross):
        raise ValueError("gross exposure must be finite")
    return max(1.0, gross)

def apply_stage_turnover_order(
    enabled,
    stage_last_rows,
    long_turnover_ups,
    short_turnover_ups,
    max_stage_inst_exposure,
    long_turnover_tolerance=None,
    short_turnover_tolerance=None,
):
    # long_turnover 让持仓往正方向变化：正仓时是开多，负仓时是平空。
    # short_turnover 让持仓往负方向变化：负仓时是开空，正仓时是平多。
    if enabled is False:
        return long_turnover_ups, short_turnover_ups
    elif enabled is True:
        long_turnover_ups = [up.copy() for up in long_turnover_ups]
        short_turnover_ups = [up.copy() for up in short_turnover_ups]
    else:
        raise ValueError(f"enforce_stage_turnover_order must be bool, got {enabled!r}")
    if long_turnover_tolerance is None:
        long_turnover_tolerance = [up.copy() for up in long_turnover_ups]
    if short_turnover_tolerance is None:
        short_turnover_tolerance = [up.copy() for up in short_turnover_ups]

    allow_open_long_by_stage = []
    allow_open_short_by_stage = []
    allow_open_long = np.ones_like(stage_last_rows[0], dtype=bool)
    allow_open_short = np.ones_like(stage_last_rows[0], dtype=bool)
    # 开仓顺序从低 stage 传到高 stage：低 stage 距离同方向满仓不超过一根 bar 容忍时，后续 stage 才能开仓。
    for stage, stage_last_row in enumerate(stage_last_rows):
        allow_open_long_by_stage.append(allow_open_long.copy())
        allow_open_short_by_stage.append(allow_open_short.copy())
        allow_open_long &= max_stage_inst_exposure - stage_last_row <= long_turnover_tolerance[stage] + EPS
        allow_open_short &= stage_last_row + max_stage_inst_exposure <= short_turnover_tolerance[stage] + EPS

    allow_close_short_by_stage = [None for _stage in stage_last_rows]
    allow_close_long_by_stage = [None for _stage in stage_last_rows]
    allow_close_short = np.ones_like(stage_last_rows[0], dtype=bool)
    allow_close_long = np.ones_like(stage_last_rows[0], dtype=bool)
    # 平仓顺序从高 stage 传到低 stage：高 stage 同方向距离平完不超过一根 bar 容忍时，前序 stage 才能平仓。
    for stage in reversed(range(len(stage_last_rows))):
        stage_last_row = stage_last_rows[stage]
        allow_close_short_by_stage[stage] = allow_close_short.copy()
        allow_close_long_by_stage[stage] = allow_close_long.copy()
        allow_close_short &= stage_last_row >= -long_turnover_tolerance[stage] - EPS
        allow_close_long &= stage_last_row <= short_turnover_tolerance[stage] + EPS

    for stage, stage_last_row in enumerate(stage_last_rows):
        open_long_allowed = allow_open_long_by_stage[stage]
        open_short_allowed = allow_open_short_by_stage[stage]
        close_short_allowed = allow_close_short_by_stage[stage]
        close_long_allowed = allow_close_long_by_stage[stage]

        long_turnover_ups[stage] = np.where((stage_last_row >= 0) & ~open_long_allowed, 0.0, long_turnover_ups[stage])
        long_turnover_ups[stage] = np.where((stage_last_row < 0) & ~close_short_allowed, 0.0, long_turnover_ups[stage])
        long_turnover_ups[stage] = np.where(
            (stage_last_row < 0) & close_short_allowed & ~open_long_allowed,
            np.minimum(long_turnover_ups[stage], -stage_last_row),
            long_turnover_ups[stage],
        )

        short_turnover_ups[stage] = np.where((stage_last_row <= 0) & ~open_short_allowed, 0.0, short_turnover_ups[stage])
        short_turnover_ups[stage] = np.where((stage_last_row > 0) & ~close_long_allowed, 0.0, short_turnover_ups[stage])
        short_turnover_ups[stage] = np.where(
            (stage_last_row > 0) & close_long_allowed & ~open_short_allowed,
            np.minimum(short_turnover_ups[stage], stage_last_row),
            short_turnover_ups[stage],
        )

    return long_turnover_ups, short_turnover_ups


def read_funding_fee_frame(_date: str, _insts):
    return None

@lru_cache(maxsize=4)
def _load_fee_cache(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype={"date": str, "inst": str})
    required = {"date", "inst", "fee_rate"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"fee cache missing columns: {sorted(missing)}")
    values = frame.set_index(["date", "inst"])["fee_rate"].astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any():
        raise ValueError("fee cache must contain finite non-negative rates")
    return values


def read_fee_rates(date: str, columns, fee_cache_csv_file) -> pd.Series:
    values = _load_fee_cache(Path(fee_cache_csv_file)).xs(date, level="date").reindex(pd.Index(columns))
    if values.isna().any():
        missing = values[values.isna()].index.tolist()
        raise KeyError(f"fee cache missing {date}: {missing[:10]}")
    return values.astype(float)


class GenPortfolio:

    def __init__(
        self,
        alpha_name,
        signal_coef,
        inst_risk_coef,
        univ_name,
        exec_info,
        max_inst_exposure,
        max_beta_exposure,
        max_open_turnover,
        max_close_turnover,
        signal_horizon,
        open_cost_coef,
        close_cost_coef,
        abnormal_corr_ban_open_thres,
        # abnormal_corr_close_thres,
        nav,
        std_name,
        req_margin,
        funding_fee_coef,
        output_holding_mode="pct",
        fee_cache=None,
    ):
        self.output_holding_mode = normalize_output_holding_mode(output_holding_mode)
        self.fee_cache = fee_cache
        assert isinstance(alpha_name, list)
        self.alpha_name = alpha_name
        self.alpha_num = len(alpha_name)

        self.stage = len(max_beta_exposure)

        self.signal_group_num = 1
        self.cost_group_num = 1
        self.signal_group_idx = None
        self.cost_group_idx = None

        assert isinstance(signal_coef, list)
        assert self.signal_group_num == len(signal_coef)
        assert self.cost_group_num == len(open_cost_coef)
        assert self.cost_group_num == len(close_cost_coef)
        self.stage = len(signal_coef[0])
        for group_signal_coef in signal_coef:
            assert isinstance(group_signal_coef, list)
            assert len(group_signal_coef) == self.stage
            for coef in group_signal_coef:
                assert isinstance(coef, list)
                assert len(coef) == self.alpha_num
            self.signal_coef = np.array(signal_coef)

        assert isinstance(inst_risk_coef, list)
        assert len(inst_risk_coef) == self.stage
        for coef in inst_risk_coef:
            assert isinstance(coef, list)
            assert len(coef) == self.alpha_num
        self.inst_risk_coef = inst_risk_coef

        self.req_margin = np.array(req_margin) * 1e-4
        self.funding_fee_coef = np.array(funding_fee_coef)

        self.abnormal_corr_ban_open_thres = np.array(abnormal_corr_ban_open_thres)
        # self.abnormal_corr_close_thres = np.array(abnormal_corr_close_thres)
        assert self.abnormal_corr_ban_open_thres.shape == (self.stage, 3)

        self.stage_oi_exposure = 0.4 / self.stage
        self.funding_fee_thres = 1e-4
        # assert len(self.abnormal_corr_close_thres) == 4

        self.univ_name = univ_name
        # self.stage_turnover_limit_rate = turnover_limit_rate / self.stage
        self.turnover_limit_rate = exec_info["turnover_limit_rate"]
        self.book_limit_rate = exec_info.get("book_limit_rate", 0.1)
        self.enforce_stage_turnover_order = require_optional_bool(exec_info, "enforce_stage_turnover_order", "strategy.params.exec_info")

        if exec_info["exec_type"] in {"make", "make2"}:
            self.exec_topk = 1
        else:
            raise ValueError(f"exec_type {exec_info['exec_type']} not supported")
        assert self.turnover_limit_rate == 0.03
        assert self.book_limit_rate == 0.1
        self.exec_type = "make"

        self.max_stage_inst_exposure = max_inst_exposure / self.stage
        self.max_beta_exposure = np.array(max_beta_exposure)
        assert self.stage == len(max_beta_exposure)
        self.max_open_turnover = np.array(max_open_turnover)
        assert self.stage == len(max_open_turnover)
        assert (self.max_open_turnover > 0).all()
        self.max_close_turnover = np.array(max_close_turnover)
        assert self.stage == len(max_close_turnover)
        assert (self.max_close_turnover > 0).all()

        self.signal_horizon = [1] + signal_horizon
        assert sorted(self.signal_horizon) == self.signal_horizon
        assert isinstance(signal_horizon, list)
        assert len(self.signal_horizon) == self.alpha_num + 1
        self.signal_horizon = np.array(self.signal_horizon)
        self.nav = nav
        self.open_cost_coef = open_cost_coef
        for group_cost in open_cost_coef:
            assert isinstance(group_cost, list)
            assert len(group_cost) == self.stage
            for coef in group_cost:
                assert isinstance(coef, list)
                assert len(coef) == self.alpha_num
        self.open_cost_coef = np.array(open_cost_coef)
        for group_cost in close_cost_coef:
            assert isinstance(group_cost, list)
            assert len(group_cost) == self.stage
            for coef in group_cost:
                assert isinstance(coef, list)
                assert len(coef) == self.alpha_num
        self.close_cost_coef = np.array(close_cost_coef)

        self.turnover_coef = (np.array(self.open_cost_coef) + np.array(self.close_cost_coef)) / 2
        self.hold_risk_coef = (np.array(self.open_cost_coef) - np.array(self.close_cost_coef)) / 2
        assert self.hold_risk_coef.flatten().min() >= -EPS
        # assert self.turnover_limit_rate == 0.02
        # self.s2_cost = 20e-4
        self.std_name = std_name

        if EXCHANGE in ["CF5m"]:
            self.max_beta_exposure_sum = 1.0
        else:
            raise ValueError(f"EXCHANGE {EXCHANGE} not supported")

        self.var_idx_new_h = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.var_idx_h_bound = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.var_idx_long_turnover = np.array([[[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.var_idx_short_turnover = np.array([[[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)] for _ in range(self.stage)])

        self.con_idx_h_bound_1 = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.con_idx_h_bound_2 = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.con_idx_turnover = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])

        self.con_idx_beta = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.con_idx_beta_sum = np.array([-1 for _ in range(self.alpha_num)])
        self.con_idx_h_sum = np.array([-1 for _ in range(self.alpha_num)])
        self.con_idx_long_turnover_sum = np.array([[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)])
        self.con_idx_short_turnover_sum = np.array([[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)])

    def build_task(
        self, insts,
        insts_values,  # 用于划分 group，可能是市值、波动率、换手率等
    ):
        n = len(insts)
        check_task_index = False # env_flag(TASK_INDEX_CHECK_ENV)

        # 生成用于划分 group 的索引
        if insts_values is None:
            insts_values = pd.Series(0, index=insts)
        insts_values = insts_values.sort_values()
        # group_idx = insts_values * 0
        group_idx = pd.Series(0, index=insts_values.index, dtype=int)
        group_size = n // self.cost_group_num
        for i in range(self.cost_group_num):
            if i != self.cost_group_num - 1:
                group_idx[insts_values.index[i * group_size:(i + 1) * group_size]] = i
            elif i == self.cost_group_num - 1:
                group_idx[insts_values.index[i * group_size:]] = i
        self.cost_group_idx = group_idx.loc[insts].values
        self.signal_group_idx = self.cost_group_idx * 0
        task = moseck_env.Task(0, 0)
        # task.putdouparam(mosek.dparam.presolve_tol_x, 1e-6)
        # task.putdouparam(mosek.dparam.basis_tol_x, 1e-4)
        # task.putdouparam(mosek.dparam.data_tol_x, 1e-4)
        # task.putdouparam(mosek.dparam.presolve_tol_primal_infeas_perturbation, 1e-4)
        # task.putdouparam(mosek.dparam.presolve_tol_rel_lindep, 1e-4)
        # task.putdouparam(mosek.dparam.presolve_tol_abs_lindep, 1e-4)
        # 自动放缩
        # task.putintparam(mosek.iparam.intpnt_scaling, mosek.iparam.scaling_free)
        # task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1.0e-3)
        task.putintparam(mosek.iparam.num_threads, 1)
        task.putintparam(mosek.iparam.log, 0)
        task.putobjsense(mosek.objsense.maximize)
        # task.appendvars(n * 3 * self.stage * self.alpha_num)
        self.stage_var_num = n * self.alpha_num * (2 + 2 * self.exec_topk)
        self.var_num = self.stage_var_num * self.stage  # 每个 stage 的变量数乘以 stage 数
        task.appendvars(self.var_num)

        # 额外每个 alpha step 都需要约束不超过 1，每个 stage 的 beta 也不超过约束
        task.appendcons((n * 3 + 1) * self.stage * self.alpha_num + self.alpha_num * 2 + self.exec_topk * 2 * n * self.alpha_num)

        var_check_list = [] if check_task_index else None
        for stage in range(self.stage):
            offset_stage = stage * self.stage_var_num
            for alpha_idx in range(self.alpha_num):
                base_for_alpha = offset_stage + (2 + self.exec_topk * 2) * n * alpha_idx
                self.var_idx_new_h[stage][alpha_idx] = base_for_alpha
                if check_task_index:
                    var_check_list += list(range(self.var_idx_new_h[stage][alpha_idx], self.var_idx_new_h[stage][alpha_idx] + n))
                self.var_idx_h_bound[stage][alpha_idx] = base_for_alpha + n
                if check_task_index:
                    var_check_list += list(range(self.var_idx_h_bound[stage][alpha_idx], self.var_idx_h_bound[stage][alpha_idx] + n))
                for i in range(self.exec_topk):
                    self.var_idx_long_turnover[stage][alpha_idx][i] = base_for_alpha + 2 * n + i * n
                    if check_task_index:
                        var_check_list += list(range(self.var_idx_long_turnover[stage][alpha_idx][i], self.var_idx_long_turnover[stage][alpha_idx][i] + n))
                    self.var_idx_short_turnover[stage][alpha_idx][i] = base_for_alpha + 2 * n + (self.exec_topk + i) * n
                    if check_task_index:
                        var_check_list += list(range(self.var_idx_short_turnover[stage][alpha_idx][i], self.var_idx_short_turnover[stage][alpha_idx][i] + n))
        
        if check_task_index:
            assert len(var_check_list) == self.var_num
            assert sorted(var_check_list) == list(range(self.var_num))

        con_check_list = [] if check_task_index else None
        curr_con_base = 0
        for stage in range(self.stage):
            offset_stage = curr_con_base + stage * 3 * self.alpha_num * n
            for alpha_idx in range(self.alpha_num):
                base_for_alpha = offset_stage + (3) * n * alpha_idx
                self.con_idx_h_bound_1[stage][alpha_idx] = base_for_alpha + 0 * n
                if check_task_index:
                    con_check_list += list(range(self.con_idx_h_bound_1[stage][alpha_idx], self.con_idx_h_bound_1[stage][alpha_idx] + n))
                self.con_idx_h_bound_2[stage][alpha_idx] = base_for_alpha + 1 * n
                if check_task_index:
                    con_check_list += list(range(self.con_idx_h_bound_2[stage][alpha_idx], self.con_idx_h_bound_2[stage][alpha_idx] + n))
                self.con_idx_turnover[stage][alpha_idx] = base_for_alpha + 2 * n
                if check_task_index:
                    con_check_list += list(range(self.con_idx_turnover[stage][alpha_idx], self.con_idx_turnover[stage][alpha_idx] + n))
        curr_con_base += n * self.stage * 3 * self.alpha_num

        for alpha_idx in range(self.alpha_num):
            self.con_idx_h_sum[alpha_idx] = curr_con_base + alpha_idx
            self.con_idx_beta_sum[alpha_idx] = curr_con_base + alpha_idx + self.alpha_num
            if check_task_index:
                con_check_list.append(self.con_idx_h_sum[alpha_idx])
                con_check_list.append(self.con_idx_beta_sum[alpha_idx])
        curr_con_base += self.alpha_num * 2

        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                self.con_idx_beta[stage][alpha_idx] = curr_con_base + stage * self.alpha_num + alpha_idx
                if check_task_index:
                    con_check_list.append(self.con_idx_beta[stage][alpha_idx])
        curr_con_base += self.stage * self.alpha_num

        for alpha_idx in range(self.alpha_num):
            base_for_alpha = curr_con_base + alpha_idx * self.exec_topk * 2 * n
            for i in range(self.exec_topk):
                base_for_i = base_for_alpha + i * n * 2
                self.con_idx_long_turnover_sum[alpha_idx][i] = base_for_i
                self.con_idx_short_turnover_sum[alpha_idx][i] = base_for_i + n
                if check_task_index:
                    con_check_list += list(range(self.con_idx_long_turnover_sum[alpha_idx][i], self.con_idx_long_turnover_sum[alpha_idx][i] + n))
                    con_check_list += list(range(self.con_idx_short_turnover_sum[alpha_idx][i], self.con_idx_short_turnover_sum[alpha_idx][i] + n))
        curr_con_base += self.alpha_num * self.exec_topk * 2 * n

        if check_task_index:
            assert len(con_check_list) == curr_con_base
            assert sorted(con_check_list) == list(range(curr_con_base))

        arange_n = np.arange(n, dtype=np.int32)
        self._zero_bound_list = [0.0] * n
        self._ra_bound_key_list = [mosek.boundkey.ra] * n
        self._fx_bound_key_list = [mosek.boundkey.fx] * n

        def index_list(start):
            return (start + arange_n).astype(np.int32, copy=False).tolist()

        self._var_idx_new_h = [[index_list(self.var_idx_new_h[stage][alpha_idx]) for alpha_idx in range(self.alpha_num)] for stage in range(self.stage)]
        self._var_idx_h_bound = [[index_list(self.var_idx_h_bound[stage][alpha_idx]) for alpha_idx in range(self.alpha_num)] for stage in range(self.stage)]
        self._var_idx_long_turnover = [
            [[index_list(self.var_idx_long_turnover[stage][alpha_idx][k]) for k in range(self.exec_topk)] for alpha_idx in range(self.alpha_num)]
            for stage in range(self.stage)
        ]
        self._var_idx_short_turnover = [
            [[index_list(self.var_idx_short_turnover[stage][alpha_idx][k]) for k in range(self.exec_topk)] for alpha_idx in range(self.alpha_num)]
            for stage in range(self.stage)
        ]
        self._con_idx_h_bound_1 = [[index_list(self.con_idx_h_bound_1[stage][alpha_idx]) for alpha_idx in range(self.alpha_num)] for stage in range(self.stage)]
        self._con_idx_h_bound_2 = [[index_list(self.con_idx_h_bound_2[stage][alpha_idx]) for alpha_idx in range(self.alpha_num)] for stage in range(self.stage)]
        self._con_idx_turnover = [[index_list(self.con_idx_turnover[stage][alpha_idx]) for alpha_idx in range(self.alpha_num)] for stage in range(self.stage)]
        self._con_idx_long_turnover_sum = [
            [index_list(self.con_idx_long_turnover_sum[alpha_idx][k]) for k in range(self.exec_topk)]
            for alpha_idx in range(self.alpha_num)
        ]
        self._con_idx_short_turnover_sum = [
            [index_list(self.con_idx_short_turnover_sum[alpha_idx][k]) for k in range(self.exec_topk)]
            for alpha_idx in range(self.alpha_num)
        ]

        # Variables
        var_bound_idx = []
        var_bound_key = []
        var_bound_low = []
        var_bound_up = []
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                var_bound_idx.extend(self._var_idx_new_h[stage][alpha_idx])
                var_bound_key.extend(self._ra_bound_key_list)
                var_bound_low.extend([-self.max_stage_inst_exposure] * n)
                var_bound_up.extend([self.max_stage_inst_exposure] * n)
                var_bound_idx.extend(self._var_idx_h_bound[stage][alpha_idx])
                var_bound_key.extend(self._ra_bound_key_list)
                var_bound_low.extend(self._zero_bound_list)
                var_bound_up.extend([self.max_stage_inst_exposure] * n)
        task.putvarboundlist(var_bound_idx, var_bound_key, var_bound_low, var_bound_up)

        con_bound_idx = []
        con_bound_key = []
        con_bound_low = []
        con_bound_up = []
        aij = []

        for stage in range(self.stage):
            # Constraints: -h_bound <= new_h <= h_bound
            for alpha_idx in range(self.alpha_num):
                con_idx = self._con_idx_h_bound_1[stage][alpha_idx]
                for j, i in enumerate(con_idx):
                    aij.extend([(i, self.var_idx_new_h[stage][alpha_idx] + j, 1), (i, self.var_idx_h_bound[stage][alpha_idx] + j, 1)])
                con_bound_idx.extend(con_idx)
                con_bound_key.extend([mosek.boundkey.lo] * n)
                con_bound_low.extend(self._zero_bound_list)
                con_bound_up.extend([float("inf")] * n)

                con_idx = self._con_idx_h_bound_2[stage][alpha_idx]
                for j, i in enumerate(con_idx):
                    aij.extend([(i, self.var_idx_new_h[stage][alpha_idx] + j, 1), (i, self.var_idx_h_bound[stage][alpha_idx] + j, -1)])
                con_bound_idx.extend(con_idx)
                con_bound_key.extend([mosek.boundkey.up] * n)
                con_bound_low.extend([-float("inf")] * n)
                con_bound_up.extend(self._zero_bound_list)

            # Constraints: -turnover <= new_h - h(t) <= turnover
            for alpha_idx in range(self.alpha_num):
                i = self.con_idx_turnover[stage][alpha_idx]
                for j in range(n):
                    if alpha_idx == 0:
                        aij += [(i + j, self.var_idx_new_h[stage][alpha_idx] + j, 1)]
                        aij += [(i + j, self.var_idx_long_turnover[stage][alpha_idx][k] + j, -1) for k in range(self.exec_topk)]
                        aij += [(i + j, self.var_idx_short_turnover[stage][alpha_idx][k] + j, 1) for k in range(self.exec_topk)]
                    elif alpha_idx != 0:
                        aij += [(i + j, self.var_idx_new_h[stage][alpha_idx] + j, 1)]
                        aij += [(i + j, self.var_idx_new_h[stage][alpha_idx - 1] + j, -1)]
                        aij += [(i + j, self.var_idx_long_turnover[stage][alpha_idx][k] + j, -1) for k in range(self.exec_topk)]
                        aij += [(i + j, self.var_idx_short_turnover[stage][alpha_idx][k] + j, 1) for k in range(self.exec_topk)]

        # Constraints: sum(h_bound) <= 1
        for alpha_idx in range(self.alpha_num):
            i = self.con_idx_h_sum[alpha_idx]
            for stage in range(self.stage):
                for j in range(n):
                    aij.append((i, self.var_idx_h_bound[stage][alpha_idx] + j, 1))
            con_bound_idx.append(i)
            con_bound_key.append(mosek.boundkey.ra)
            con_bound_low.append(0.0)
            con_bound_up.append(1.0)

        # beta is fixed at 1 for every instrument, so these coefficients are static.
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                i = self.con_idx_beta[stage][alpha_idx]
                for j in range(n):
                    aij.append((i, self.var_idx_new_h[stage][alpha_idx] + j, 1))

        for alpha_idx in range(self.alpha_num):
            i = self.con_idx_beta_sum[alpha_idx]
            for stage in range(self.stage):
                for j in range(n):
                    aij.append((i, self.var_idx_new_h[stage][alpha_idx] + j, 1))

        # Constraints: sum(stage_turnover) <= turnover_limit
        for alpha_idx in range(self.alpha_num):
            for k in range(self.exec_topk):
                i = self.con_idx_long_turnover_sum[alpha_idx][k]
                for stage in range(self.stage):
                    for j in range(n):
                        aij.append((i + j, self.var_idx_long_turnover[stage][alpha_idx][k] + j, 1))

                i = self.con_idx_short_turnover_sum[alpha_idx][k]
                for stage in range(self.stage):
                    for j in range(n):
                        aij.append((i + j, self.var_idx_short_turnover[stage][alpha_idx][k] + j, 1))

        task.putconboundlist(con_bound_idx, con_bound_key, con_bound_low, con_bound_up)
        task.putaijlist(*zip(*aij))
        return task
    
    def pct_lines_to_output_lines(self, pct_lines, close, valid=None, original_lines=None):
        if self.output_holding_mode == "qty":
            valid_line = None
            if valid is not None:
                valid_line = pd.Series(valid, index=pct_lines[0].index, dtype=bool)
            qty_lines = [
                pct_line_to_qty(
                    line, close, self.nav, valid=valid_line,
                    previous_qty_line=original_lines[idx] if original_lines is not None else None,
                ) for idx, line in enumerate(pct_lines)
            ]
            return qty_lines
        if self.output_holding_mode == "pct":
            return pct_lines

        raise ValueError(f"unknown output_holding_mode {self.output_holding_mode!r}")

    def update_one_line(
        self,
        task,
        date,
        last_row,
        previous_close,
        previous_output_row,
        close,
        alphas,
        std,
        valid,
        fee,
        short_price,
        short_vol,
        long_price,
        long_vol,
        corr1,
        corr2,
        corr3,
        oi,
        funding_fee,
        timestamp,
        return_state=False,
    ):
        # Optimizer state is always pct of nav. Pct自然飘逸使用 close / previous_close，
        # 避免依赖 ret1m 的采样口径和缺失处理。
        valid = pd.Series(valid, index=last_row[0].index, dtype=bool)
        last_row, next_previous_close = drift_pct_lines_by_close(last_row, close, previous_close, valid)
        gross_exposure_bound = current_gross_exposure_bound(last_row)
        if not valid.any():
            output_lines = self.pct_lines_to_output_lines(last_row, close, valid, previous_output_row)
            if return_state:
                return output_lines, last_row, next_previous_close
            return output_lines

        n = len(last_row[0])
        universe = last_row[0].index
        last_row_values = [line.values for line in last_row]

        alphas = np.array(alphas)
        if oi is None:
            oi = np.full(n, np.inf)
        if funding_fee is None:
            funding_fee = np.zeros(n)
        oi_exposure_ratio = np.clip(oi * self.stage_oi_exposure / self.nav / self.max_stage_inst_exposure, 0, 1)
        turnover_scale = self.signal_horizon * self.turnover_limit_rate / self.nav

        mid_price = (short_price[0] + long_price[0]) / 2
        
        long_cost = np.array([(long_price[k] / mid_price - 1) + fee + self.req_margin for k in range(self.exec_topk)])
        short_cost = np.array([-(short_price[k] / mid_price - 1) + fee + self.req_margin for k in range(self.exec_topk)])
        adjust_hold_risk = fee + (long_price[0] - short_price[0]) / mid_price / 2

        clip_funding_fee = np.sign(funding_fee) * np.clip(np.abs(funding_fee) - self.funding_fee_thres, 0, np.inf)
        long_cost = np.clip(long_cost + clip_funding_fee * self.funding_fee_coef, 0, np.inf)
        short_cost = np.clip(short_cost + clip_funding_fee * self.funding_fee_coef, 0, np.inf)

        zero_bound_list = self._zero_bound_list
        ra_bound_key_list = self._ra_bound_key_list
        objective_idx = []
        objective_val = []
        var_bound_idx = []
        var_bound_key = []
        var_bound_low = []
        var_bound_up = []
        con_bound_idx = []
        con_bound_key = []
        con_bound_low = []
        con_bound_up = []

        # Objective
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                new_h_coef = alphas[alpha_idx] * self.signal_coef[self.signal_group_idx, stage, alpha_idx]
                h_bound_coef = -(std * std * self.inst_risk_coef[stage][alpha_idx]) - adjust_hold_risk * self.hold_risk_coef[self.cost_group_idx, stage, alpha_idx]
                objective_idx.extend(self._var_idx_new_h[stage][alpha_idx])
                objective_val.extend(np.where(valid, new_h_coef, 0.0).tolist())
                objective_idx.extend(self._var_idx_h_bound[stage][alpha_idx])
                objective_val.extend(np.where(valid, h_bound_coef, 0.0).tolist())
                for k in range(self.exec_topk):
                    long_turnover_coef = -long_cost[k] * self.turnover_coef[self.cost_group_idx, stage, alpha_idx]
                    short_turnover_coef = -short_cost[k] * self.turnover_coef[self.cost_group_idx, stage, alpha_idx]
                    objective_idx.extend(self._var_idx_long_turnover[stage][alpha_idx][k])
                    objective_val.extend(np.where(valid, long_turnover_coef, 0.0).tolist())
                    objective_idx.extend(self._var_idx_short_turnover[stage][alpha_idx][k])
                    objective_val.extend(np.where(valid, short_turnover_coef, 0.0).tolist())
        task.putclist(objective_idx, objective_val)

        # Variables
        long_limit = np.array([long_price[k] * long_vol[k] for k in range(self.exec_topk)])
        short_limit = np.array([short_price[k] * short_vol[k] for k in range(self.exec_topk)])
        long_limit = np.where(valid, long_limit, 0.0)
        short_limit = np.where(valid, short_limit, 0.0)
        valid_bound_key = [mosek.boundkey.ra if is_valid else mosek.boundkey.fx for is_valid in valid]
        long_turnover_ups = [[[None for _ in range(self.exec_topk)] for _ in range(self.alpha_num)] for _ in range(self.stage)]
        short_turnover_ups = [[[None for _ in range(self.exec_topk)] for _ in range(self.alpha_num)] for _ in range(self.stage)]
        for alpha_idx in range(self.alpha_num):
            for k in range(self.exec_topk):
                stage_long_turnover_ups = []
                stage_short_turnover_ups = []
                for stage, stage_last_row in enumerate(last_row_values):
                    long_turnover_cap = np.where(stage_last_row >= 0, self.max_open_turnover[stage], self.max_close_turnover[stage])
                    short_turnover_cap = np.where(stage_last_row <= 0, self.max_open_turnover[stage], self.max_close_turnover[stage])
                    long_volume_limit = long_limit[k] * turnover_scale[alpha_idx]
                    short_volume_limit = short_limit[k] * turnover_scale[alpha_idx]
                    stage_long_turnover_ups.append(np.minimum(long_volume_limit, long_turnover_cap * oi_exposure_ratio))
                    stage_short_turnover_ups.append(np.minimum(short_volume_limit, short_turnover_cap * oi_exposure_ratio))
                # 先按原逻辑得到逐合约成交上界，再可选应用 stage 顺序限制。
                stage_long_turnover_ups, stage_short_turnover_ups = apply_stage_turnover_order(
                    self.enforce_stage_turnover_order,
                    last_row_values,
                    stage_long_turnover_ups,
                    stage_short_turnover_ups,
                    self.max_stage_inst_exposure,
                )
                for stage in range(self.stage):
                    long_turnover_ups[stage][alpha_idx][k] = stage_long_turnover_ups[stage]
                    short_turnover_ups[stage][alpha_idx][k] = stage_short_turnover_ups[stage]
        for stage in range(self.stage):
            stage_last_row = last_row_values[stage]
            max_stage_inst_exposure = np.maximum(self.max_stage_inst_exposure, np.abs(stage_last_row))
            for alpha_idx in range(self.alpha_num):
                var_bound_idx.extend(self._var_idx_new_h[stage][alpha_idx])
                var_bound_key.extend(ra_bound_key_list)
                var_bound_low.extend((-max_stage_inst_exposure).tolist())
                var_bound_up.extend(max_stage_inst_exposure.tolist())
                var_bound_idx.extend(self._var_idx_h_bound[stage][alpha_idx])
                var_bound_key.extend(ra_bound_key_list)
                var_bound_low.extend(zero_bound_list)
                var_bound_up.extend(max_stage_inst_exposure.tolist())

                for k in range(self.exec_topk):
                    long_turnover_up = long_turnover_ups[stage][alpha_idx][k]
                    short_turnover_up = short_turnover_ups[stage][alpha_idx][k]
                    var_bound_idx.extend(self._var_idx_long_turnover[stage][alpha_idx][k])
                    var_bound_key.extend(valid_bound_key)
                    var_bound_low.extend(zero_bound_list)
                    var_bound_up.extend(np.where(valid, long_turnover_up, 0.0).tolist())
                    var_bound_idx.extend(self._var_idx_short_turnover[stage][alpha_idx][k])
                    var_bound_key.extend(valid_bound_key)
                    var_bound_low.extend(zero_bound_list)
                    var_bound_up.extend(np.where(valid, short_turnover_up, 0.0).tolist())
        task.putvarboundlist(var_bound_idx, var_bound_key, var_bound_low, var_bound_up)

        # Constraints: sum(stage_turnover) <= turnover_limit
        for alpha_idx in range(self.alpha_num):
            con_bound_idx.append(self.con_idx_h_sum[alpha_idx])
            con_bound_key.append(mosek.boundkey.ra)
            con_bound_low.append(0.0)
            con_bound_up.append(gross_exposure_bound)
            for k in range(self.exec_topk):
                con_bound_idx.extend(self._con_idx_long_turnover_sum[alpha_idx][k])
                con_bound_key.extend(ra_bound_key_list)
                con_bound_low.extend(zero_bound_list)
                con_bound_up.extend((long_limit[k] * turnover_scale[alpha_idx]).tolist())

                con_bound_idx.extend(self._con_idx_short_turnover_sum[alpha_idx][k])
                con_bound_key.extend(ra_bound_key_list)
                con_bound_low.extend(zero_bound_list)
                con_bound_up.extend((short_limit[k] * turnover_scale[alpha_idx]).tolist())

        # Constraints
        for stage in range(self.stage):
            stage_last_row = last_row_values[stage]
            for alpha_idx in range(self.alpha_num):
                con_bound_idx.extend(self._con_idx_turnover[stage][alpha_idx])
                con_bound_key.extend(self._fx_bound_key_list)
                if alpha_idx == 0:
                    con_bound_low.extend(stage_last_row.tolist())
                    con_bound_up.extend(stage_last_row.tolist())
                elif alpha_idx != 0:
                    con_bound_low.extend(zero_bound_list)
                    con_bound_up.extend(zero_bound_list)

        # beta is fixed at 1 for every instrument, so beta exposure is net exposure.
        curr_stage_beta = [stage_last_row.sum() for stage_last_row in last_row_values]
        curr_beta = sum(curr_stage_beta)

        # Constraints: -max_beta_exposure <= sum(new_h) <= max_beta_exposure
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                i = self.con_idx_beta[stage][alpha_idx]
                beta_exposure = max(self.max_beta_exposure[stage], abs(curr_stage_beta[stage]))
                con_bound_idx.append(i)
                con_bound_key.append(mosek.boundkey.ra)
                con_bound_low.append(-beta_exposure)
                con_bound_up.append(beta_exposure)

        beta_constraint = max(self.max_beta_exposure_sum, abs(curr_beta))
        for alpha_idx in range(self.alpha_num):
            i = self.con_idx_beta_sum[alpha_idx]
            con_bound_idx.append(i)
            con_bound_key.append(mosek.boundkey.ra)
            con_bound_low.append(-beta_constraint)
            con_bound_up.append(beta_constraint)
        task.putconboundlist(con_bound_idx, con_bound_key, con_bound_low, con_bound_up)

        task.optimize()
        solsta = task.getsolsta(mosek.soltype.itr)

        if solsta == mosek.solsta.optimal:
            x = np.zeros(self.var_num)
            task.getxx(mosek.soltype.itr, x)
            line_list = [x[stage * self.stage_var_num:stage * self.stage_var_num + n] for stage in range(self.stage)]
            fix_line_list = [pd.Series(line, index=universe) for line in line_list]
            output_lines = self.pct_lines_to_output_lines(fix_line_list, close, valid, previous_output_row)
            if return_state:
                return output_lines, fix_line_list, next_previous_close
            return output_lines

        if solsta != mosek.solsta.optimal:
            print(f"solsta = {solsta} at {timestamp}")
            fix_line_list = [pd.Series(line, index=universe) for line in last_row]
            output_lines = self.pct_lines_to_output_lines(fix_line_list, close, valid, previous_output_row)
            if return_state:
                return output_lines, fix_line_list, next_previous_close
            return output_lines

    def update_one_day(
        self,
        date,
        last_row,
        previous_close,
        previous_output_row,
        alphas,
        std,
        fee,
        close,
        short_price,
        short_vol,
        long_price,
        long_vol,
        corr1,
        corr2,
        corr3,
        oi,
        funding_fee,
        valid,
    ):
        task = self.build_task(alphas[0].columns, None)

        holding = pd.DataFrame(0, columns=alphas[0].columns, index=alphas[0].index, dtype=float)
        alpha_values = [alpha.values for alpha in alphas]
        std_values = std.values
        close_values = close.values
        valid_values = valid.values
        fee_values = fee.values
        short_price_values = [bp.values for bp in short_price]
        short_vol_values = [bv.values for bv in short_vol]
        long_price_values = [ap.values for ap in long_price]
        long_vol_values = [av.values for av in long_vol]

        def optional_values(frame):
            if frame is None:
                return None
            return frame.values

        corr1_values = optional_values(corr1)
        corr2_values = optional_values(corr2)
        corr3_values = optional_values(corr3)
        oi_values = optional_values(oi)
        funding_fee_values = optional_values(funding_fee)

        def optional_row_values(values, row_idx):
            if values is None:
                return None
            return values[row_idx, :]

        last_output_row = previous_output_row
        for i, ts in enumerate(alphas[0].index):
            output_row, last_row, previous_close = self.update_one_line(
                task=task,
                date=date,
                last_row=last_row,
                previous_close=previous_close,
                previous_output_row=last_output_row,
                close=close_values[i, :],
                alphas=[alpha_value[i, :] for alpha_value in alpha_values],
                std=std_values[i, :],
                valid=valid_values[i, :],
                fee=fee_values,
                short_price=[bp[i, :] for bp in short_price_values],
                short_vol=[bv[i, :] for bv in short_vol_values],
                long_price=[ap[i, :] for ap in long_price_values],
                long_vol=[av[i, :] for av in long_vol_values],
                corr1=optional_row_values(corr1_values, i),
                corr2=optional_row_values(corr2_values, i),
                corr3=optional_row_values(corr3_values, i),
                oi=optional_row_values(oi_values, i),
                funding_fee=optional_row_values(funding_fee_values, i),
                timestamp=ts,
                return_state=True,
            )
            holding.iloc[i] = sum(output_row)
            last_output_row = output_row
        return holding, last_row, previous_close, last_output_row

    
    def get_holding(self, dates):
        # prepare
        holdings = []
        pbar = tqdm(total=len(dates))
        last_row = [pd.Series(0.0, index=[]) for _ in range(self.stage)]
        previous_close = pd.Series(dtype=float)
        last_output_row = [pd.Series(0.0, index=[]) for _ in range(self.stage)]

        for date in dates:
            try:
                insts = read_universe(date, self.univ_name).index
                today_index = get_index(date)
                alphas = [read_signal(date, self.univ_name, alpha_name) for alpha_name in self.alpha_name]

                for alpha in alphas:
                    assert alpha.columns.equals(insts)
                    assert np.isfinite(alpha.values.flatten()).all()
                    assert (alpha.index == today_index).all()

                basedata = [read_basedata(date, inst) for inst in insts]
                if EXCHANGE in ["CF5m"]:
                    book1_value_sum0 = pd.DataFrame(np.inf, index=today_index, columns=insts)
                else:
                    raise ValueError(f"EXCHANGE {EXCHANGE} not supported")

                adjust_book1_value_sum0 = book1_value_sum0 / self.turnover_limit_rate * self.book_limit_rate
                close = pd.concat([bd["close"].rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1)

                if self.exec_type == "make":
                    half_spread_mean = read_feature_frame(date, insts, "half_spread_mean").fillna(0.0)
                    short_price = [close - half_spread_mean]
                    long_price = [close + half_spread_mean]
                else:
                    raise ValueError(f"exec_type {self.exec_type} not supported")
                
                turnover = pd.concat([bd["turnover"].rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1)
                turnover = turnover.where(turnover < adjust_book1_value_sum0, adjust_book1_value_sum0)
                short_vol = [turnover / short_price[k] / self.exec_topk for k in range(self.exec_topk)]
                long_vol = [turnover / long_price[k] / self.exec_topk for k in range(self.exec_topk)]
                last_row = [l.reindex(insts).fillna(0.0) for l in last_row]
                previous_close = previous_close.reindex(insts)
                last_output_row = [l.reindex(insts).fillna(0.0) for l in last_output_row]
                
                if EXCHANGE in ["CF5m"]:
                    # self.fee_cache is a csv fee file
                    fee = read_fee_rates(date, insts, self.fee_cache)
                else:
                    raise ValueError(f"EXCHANGE {EXCHANGE} not supported")

                valid = read_valid_frame(date, insts)
                std = read_feature_frame(date, insts, self.std_name)
                corr1 = None
                corr2 = None
                corr3 = None
                oi = None
                funding_fee = read_funding_fee_frame(date, insts)

                holding, last_row, previous_close, last_output_row = self.update_one_day(
                    date=date,
                    last_row=last_row,
                    previous_close=previous_close,
                    previous_output_row=last_output_row,
                    alphas=alphas,
                    std=std,
                    fee=fee,
                    close=close,
                    short_price=short_price,
                    short_vol=short_vol,
                    long_price=long_price,
                    long_vol=long_vol,
                    corr1=corr1,
                    corr2=corr2,
                    corr3=corr3,
                    oi=oi,
                    funding_fee=funding_fee,
                    valid=valid,
                )
                holdings.append(holding)
            except Exception as exc:
                raise RuntimeError(f"error at {date} {exc}") from exc
            pbar.update()

        return holdings     

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    # parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def main():
    args = parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = json5.load(handle)
    start_date = args.start_date or config["start_date"]
    end_date = args.end_date or config["end_date"]
    
    dates = get_dates(start_date, end_date)

    params = dict(config["strategy"]["params"])
    strategy = GenPortfolio(
        **params,
        alpha_name=config["alpha_name"],
        univ_name=config["univ_name"],
        nav=config["nav"],
        std_name=config["std_name"],
        fee_cache=config.get("fee_cache"),
    )
    holdings = strategy.get_holding(dates)
    for date, holding in zip(dates, holdings):
        write_holding(date, config["univ_name"], config["out_name"], holding)
        print(f"{date} holding written")


if __name__ == "__main__":
    main()
