import os
import sys
import time
import traceback
import json5
import mosek
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Pool
from time import sleep

from sphinx.util.exchange_api import get_index, read_alpha, read_basedata, read_fee, read_orderbook, read_signal, read_universe
from sphinx.util.runtime_config import get_env_exchange


EXCHANGE = get_env_exchange()

FEE = np.inf

# import warnings

# # 忽略所有警告
# warnings.filterwarnings('ignore')

os.environ["MOSEKLM_LICENSE_FILE"] = "mosek.lic"
env = mosek.Env()

# Objective
#       Maximize:
#           new_h * alpha(t) - sum(turnover * cost1) - sum(h_bound * risk_hold_coef)
#           +
#          new_h_s2 * alpha(t) - sum(turnover_s2 * cost1_s2) - sum(h_bound_s2 * risk_hold_coef)
#           +
#           new_h2 * alpha2(t) - sum(turnover2 * cost2) - sum(h_bound2 * risk_hold_coef2)
#           +
#           new_h2_s2 * alpha2(t) - sum(turnover2_s2 * cost2_s2) - sum(h_bound2_s2 * risk_hold_coef2)
# Variables
#       x = [new_h, turnover, h_bound, new_h2, turnover2, h_hound2, new_h_s2, turnover_s2, h_bound_s2, new_h2_s2, turnover2_s2, h_hound2_s2] (dimension is 6 * n * 2, percentage of nav)
#       turnover <= turnover_limit(t) * turnover_limit_rate / nav
#       turnover_s2 <= turnover_limit(t) * turnover_limit_rate_s2 / nav
#       turnover2 <= turnover_limit(t) * turnover_limit_rate / nav
#       turnover2 <= turnover_limit(t) * turnover_limit_rate_s2 / nav
#       h_bound <= max_inst_exposure
#       h_bound_s2 <= max_inst_exposure_s2
#       h_bound2 <= max_inst_exposure
#       h_bound2_s2 <= max_inst_exposure_s2
#
# Constraints
#       -h_bound <= new_h <= h_bound
#       0:n       new_h + h_bound >=0
#       n:2n      new_h - h_bound <=0
#
#       -turnover <= new_h - h(t) <= turnover
#       2n:3n     new_h + turnover >= h(t)
#       3n:4n     new_h - turnover <= h(t)
#
#       -h_bound2 <= new_h2 <= h_bound2
#       4n:5n     new_h2 + h_bound2 >= 0
#       5n:6n     new_h2 - h_bound2 <= 0
#
#       -turnover2 <= new_h2 - h2(t) <= turnover2
#       6n:7n     new_h2 + turnover2 >= new_h
#       7n:8n     new_h2 - turnover2 <= new_h
#
#       -h_bound_s2 <= new_h_s2 <= h_bound_s2
#       8n:9n     new_h_s2 + h_bound_s2 >=0
#       9n:10n    new_h_s2 - h_bound_s2 <=0
#
#       -turnover_s2 <= new_h_s2 - h_s2(t) <= turnover_s2
#       10n:11n   new_h_s2 + turnover_s2 >= h_s2(t)
#       11n:12n   new_h_s2 - turnover_s2 <= h_s2(t)
#
#       -h_bound2_s2 <= new_h_s2 <= h_bound2_s2
#       12n:13n   new_h2_s2 + h_bound2_s2 >= 0
#       13n:14n   new_h2_s2 - h_bound2_s2 <= 0
#
#       -turnover2_s2 <= new_h2_s2 - h2_s2(t) <= turnover2_s2
#       14n:15n   new_h2_s2 + turnover2_s2 >= new_h_s2
#       15n:16n   new_h2_s2 - turnover2_s2 <= new_h_s2
#
#       16n       sum(h_bound) + sum(h_bound_s2) <= 1
#       16n+1     sum(h_bound2) + sum(h_hound2_s2) <= 1
#       16n+2     -max_beta_exposure <= sum_beta(new_h) + sum_beta(new_h_s2) <= max_beta_exposure
#       16n+3     -max_beta_exposure <= sum_beta(new_h2) + sum_beta(new_h2_s2) <= max_beta_exposure
#
# Where
#       n is number of futures

EPS = 1e-7


class GenPortfolio:

    def __init__(
        self,
        alpha_name,
        prob_name,
        signal_coef,
        # open_prob_thres,
        # close_prob_thres,
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
        # abnormal_turnver_ban_thres1,
        # abnormal_turnver_ban_thres2,
        # abnormal_ret1m_ban_thres,
        abnormal_turnver_ban_open_beta,
        abnormal_corr_ban_open_thres,
        # abnormal_corr_close_thres,
        nav,
        std_name,  # 被同时用于计算 risk 和划分 group
        req_margin,
        funding_fee_coef,
    ):
        assert isinstance(alpha_name, list)
        self.alpha_name = alpha_name
        self.alpha_num = len(alpha_name)

        assert isinstance(prob_name, list)
        self.prob_name = prob_name
        assert len(prob_name) == self.alpha_num

        self.stage = len(max_beta_exposure)  # 开仓分为 4 个阶段，每个阶段分别打破一次 turnover limit 和 max inst exposure

        self.signal_group_num = 1  # 把合约用市值、波动率、换手率等指标分为 3 组，每组用不同的参数，由于太麻烦，暂时只用不同的 signal_coef
        self.cost_group_num = 1  # 把合约用市值、波动率、换手率等指标分为 3 组，每组用不同的参数，由于太麻烦，暂时只用不同的 signal_coef
        self.signal_group_idx = None
        self.cost_group_idx = None

        assert isinstance(signal_coef, list)
        assert self.signal_group_num == len(signal_coef)
        assert self.cost_group_num == len(open_cost_coef)
        assert self.cost_group_num == len(close_cost_coef)
        self.stage = len(signal_coef[0])  # 开仓分为 4 个阶段，每个阶段分别打破一次 turnover limit 和 max inst exposure
        for group_signal_coef in signal_coef:
            assert isinstance(group_signal_coef, list)
            assert len(group_signal_coef) == self.stage
            for coef in group_signal_coef:
                assert isinstance(coef, list)
                assert len(coef) == self.alpha_num
            self.signal_coef = np.array(signal_coef)

        # assert isinstance(open_prob_thres, list)
        # assert len(open_prob_thres) == self.stage
        # for coef in open_prob_thres:
        #     assert isinstance(coef, list)
        #     assert len(coef) == self.alpha_num
        # self.open_prob_thres = np.array(open_prob_thres)

        # assert isinstance(close_prob_thres, list)
        # assert len(close_prob_thres) == self.stage
        # for coef in close_prob_thres:
        #     assert isinstance(coef, list)
        #     assert len(coef) == self.alpha_num
        # self.close_prob_thres = np.array(close_prob_thres)

        assert isinstance(inst_risk_coef, list)
        assert len(inst_risk_coef) == self.stage
        for coef in inst_risk_coef:
            assert isinstance(coef, list)
            assert len(coef) == self.alpha_num
        self.inst_risk_coef = inst_risk_coef

        self.req_margin = np.array(req_margin) * 1e-4
        self.funding_fee_coef = np.array(funding_fee_coef)

        # assert isinstance(abnormal_turnver_ban_thres1, list)
        # assert isinstance(abnormal_turnver_ban_thres2, list)
        # assert len(abnormal_turnver_ban_thres1) == self.stage
        # assert len(abnormal_turnver_ban_thres2) == self.stage
        # for thres in abnormal_turnver_ban_thres1:
        #     assert isinstance(thres, list)
        #     assert len(thres) == 2
        # for thres in abnormal_turnver_ban_thres2:
        #     assert isinstance(thres, list)
        #     assert len(thres) == 2

        assert isinstance(abnormal_turnver_ban_open_beta, list)
        assert len(abnormal_turnver_ban_open_beta) == self.alpha_num
        self.abnormal_turnver_ban_open_beta = np.array(abnormal_turnver_ban_open_beta)

        # self.abnormal_turnver_too_large_quick_thres = np.array([i[0] for i in abnormal_turnver_ban_thres1])
        # self.abnormal_turnver_too_small_quick_thres = np.array([i[1] for i in abnormal_turnver_ban_thres1])
        # assert (self.abnormal_turnver_too_large_quick_thres > self.abnormal_turnver_too_small_quick_thres).all()
        # self.abnormal_turnver_too_large_slow_thres = np.array([i[0] for i in abnormal_turnver_ban_thres2])
        # self.abnormal_turnver_too_small_slow_thres = np.array([i[1] for i in abnormal_turnver_ban_thres2])
        # assert (self.abnormal_turnver_too_large_slow_thres > self.abnormal_turnver_too_small_slow_thres).all()
        # self.abnormal_turnver_ban_long_thres1 = np.array([i[0] for i in abnormal_turnver_ban_thres1])
        # self.abnormal_turnver_ban_short_thres1 = np.array([i[1] for i in abnormal_turnver_ban_thres1])
        # self.abnormal_turnver_ban_long_thres2 = np.array([i[0] for i in abnormal_turnver_ban_thres2])
        # self.abnormal_turnver_ban_short_thres2 = np.array([i[1] for i in abnormal_turnver_ban_thres2])

        self.abnormal_corr_ban_open_thres = np.array(abnormal_corr_ban_open_thres)
        # self.abnormal_corr_close_thres = np.array(abnormal_corr_close_thres)
        assert self.abnormal_corr_ban_open_thres.shape == (self.stage, 3)

        self.stage_oi_exposure = 0.4 / self.stage
        self.funding_fee_thres = 1e-4
        # assert len(self.abnormal_corr_close_thres) == 4
        # assert isinstance(abnormal_ret1m_ban_thres, list)
        # assert len(abnormal_ret1m_ban_thres) == self.stage
        # for thres in abnormal_ret1m_ban_thres:
        #     assert isinstance(thres, list)
        #     assert len(thres) == 2
        # self.abnormal_ret1m_ban_long_thres = np.array([i[0] for i in abnormal_ret1m_ban_thres])
        # self.abnormal_ret1m_ban_short_thres = np.array([i[1] for i in abnormal_ret1m_ban_thres])
        # 暂时不要用 req_margin，因为可能是过拟合严重一些，okx 样本外 20250203 的表现不好回撤较大
        # assert isinstance(req_margin, list)
        # assert len(req_margin) == self.stage
        # self.req_margin = np.array(req_margin)

        self.univ_name = univ_name
        # self.stage_turnover_limit_rate = turnover_limit_rate / self.stage
        self.turnover_limit_rate = exec_info["turnover_limit_rate"]
        self.book_limit_rate = exec_info["book_limit_rate"]

        if exec_info["exec_type"] == "take":
            self.exec_topk = exec_info["topk"]
            # assert turnover_limit_rate == 0.5  # 资管自营各一半
            # assert turnover_limit_rate == 0.5  # 假设吃到一半，别乱动，要配合滑点估计
            # assert turnover_limit_rate == 1.0  # 期货暂时只有一个账户
            # assert self.exec_topk == 3
            # assert turnover_limit_rate == 1.0
            assert self.exec_topk == 1
        elif exec_info["exec_type"] == "take_bp":
            self.exec_topk = exec_info["topk"]
            # assert self.exec_topk == 1
        elif exec_info["exec_type"] == "make":
            self.exec_topk = len(exec_info["extra_slippage"])
            self.extra_slippage = np.array(exec_info["extra_slippage"])
            assert self.turnover_limit_rate == 0.03
            assert self.book_limit_rate == 0.1
        elif exec_info["exec_type"] == "make2":
            self.exec_topk = 1
            assert self.turnover_limit_rate == 0.03
            assert self.book_limit_rate == 0.1
        else:
            raise ValueError(f"exec_type {exec_info['exec_type']} not supported")
        self.exec_type = exec_info["exec_type"]

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
        # 增加开仓成本，减少平仓成本，否则低信号时会有因为平仓阈值过大而持有大量无意义的仓位带来波动
        # 交易成本较高时，这个比值可能会需要调整
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

        # assert len(self.open_cost_coef) == self.stage
        # for coef in self.open_cost_coef:
        #     assert isinstance(coef, list)
        #     assert len(coef) == self.alpha_num
        # self.open_cost_coef = np.array(self.open_cost_coef)
        # self.close_cost_coef = close_cost_coef
        # assert len(self.close_cost_coef) == self.stage
        # for coef in self.close_cost_coef:
        #     assert isinstance(coef, list)
        #     assert len(coef) == self.alpha_num
        # self.close_cost_coef = np.array(self.close_cost_coef)
        # self.hold_risk_coef = (self.open_close_cost_ratio - 1) / (self.open_close_cost_ratio + 1)
        # self.turnover_coef = (self.open_close_cost_ratio[0] + self.open_close_cost_ratio[1]) / 2
        # self.hold_risk_coef = (self.open_close_cost_ratio[0] - self.open_close_cost_ratio[1]) / 2
        self.turnover_coef = (np.array(self.open_cost_coef) + np.array(self.close_cost_coef)) / 2
        self.hold_risk_coef = (np.array(self.open_cost_coef) - np.array(self.close_cost_coef)) / 2
        assert self.hold_risk_coef.flatten().min() >= -EPS
        # assert self.turnover_limit_rate == 0.02
        # self.s2_cost = 20e-4
        self.std_name = std_name

        if EXCHANGE in ["okx10m", "okx5m"]:
            # self.max_beta_exposure_sum = 0.05
            self.max_beta_exposure_sum = 0.5
        elif EXCHANGE in ["binance5m"]:
            self.max_beta_exposure_sum = 0.5
        elif EXCHANGE in ["CF5m"]:
            self.max_beta_exposure_sum = 1.0
        else:
            raise ValueError(f"EXCHANGE {EXCHANGE} not supported")
        self.today_beta = np.array([])

        self.val_idx_new_h = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.val_idx_h_bound = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.val_idx_long_turnover = np.array([[[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.val_idx_short_turnover = np.array([[[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)] for _ in range(self.stage)])

        self.constrain_idx_h_bound_1 = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.constrain_idx_h_bound_2 = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.constrain_idx_turnover = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])

        self.constrain_idx_beta = np.array([[-1 for _ in range(self.alpha_num)] for _ in range(self.stage)])
        self.constrain_idx_beta_sum = np.array([-1 for _ in range(self.alpha_num)])
        self.constrain_idx_h_sum = np.array([-1 for _ in range(self.alpha_num)])
        self.constrain_idx_long_turnover_sum = np.array([[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)])
        self.constrain_idx_short_turnover_sum = np.array([[-1 for _ in range(self.exec_topk)] for _ in range(self.alpha_num)])

    def build_task(
        self,
        insts,
        insts_values,  # 用于划分 group，可能是市值、波动率、换手率等
    ):
        n = len(insts)

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
            else:
                group_idx[insts_values.index[i * group_size:]] = i
        self.cost_group_idx = group_idx.loc[insts].values
        self.signal_group_idx = self.cost_group_idx * 0
        task = env.Task(0, 0)
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

        self.today_beta = np.ones(n)

        val_check_list = []
        for stage in range(self.stage):
            offset_stage = stage * self.stage_var_num
            for alpha_idx in range(self.alpha_num):
                base_for_alpha = offset_stage + (2 + self.exec_topk * 2) * n * alpha_idx
                self.val_idx_new_h[stage][alpha_idx] = base_for_alpha
                val_check_list += list(range(self.val_idx_new_h[stage][alpha_idx], self.val_idx_new_h[stage][alpha_idx] + n))
                self.val_idx_h_bound[stage][alpha_idx] = base_for_alpha + n
                val_check_list += list(range(self.val_idx_h_bound[stage][alpha_idx], self.val_idx_h_bound[stage][alpha_idx] + n))
                for i in range(self.exec_topk):
                    self.val_idx_long_turnover[stage][alpha_idx][i] = base_for_alpha + 2 * n + i * n
                    val_check_list += list(range(self.val_idx_long_turnover[stage][alpha_idx][i], self.val_idx_long_turnover[stage][alpha_idx][i] + n))
                    self.val_idx_short_turnover[stage][alpha_idx][i] = base_for_alpha + 2 * n + (self.exec_topk + i) * n
                    val_check_list += list(range(self.val_idx_short_turnover[stage][alpha_idx][i], self.val_idx_short_turnover[stage][alpha_idx][i] + n))
        # assert len(val_check_list) == self.stage * self.alpha_num * (2 + 2 * self.exec_topk) * n
        # assert sorted(val_check_list) == list(range(len(val_check_list)))

        constrain_check_list = []
        curr_constrain_base = 0
        for stage in range(self.stage):
            offset_stage = curr_constrain_base + stage * 3 * self.alpha_num * n
            for alpha_idx in range(self.alpha_num):
                base_for_alpha = offset_stage + (3) * n * alpha_idx
                self.constrain_idx_h_bound_1[stage][alpha_idx] = base_for_alpha + 0 * n
                constrain_check_list += list(range(self.constrain_idx_h_bound_1[stage][alpha_idx], self.constrain_idx_h_bound_1[stage][alpha_idx] + n))
                self.constrain_idx_h_bound_2[stage][alpha_idx] = base_for_alpha + 1 * n
                constrain_check_list += list(range(self.constrain_idx_h_bound_2[stage][alpha_idx], self.constrain_idx_h_bound_2[stage][alpha_idx] + n))
                self.constrain_idx_turnover[stage][alpha_idx] = base_for_alpha + 2 * n
                constrain_check_list += list(range(self.constrain_idx_turnover[stage][alpha_idx], self.constrain_idx_turnover[stage][alpha_idx] + n))
        curr_constrain_base += n * self.stage * 3 * self.alpha_num
        # assert len(constrain_check_list) == curr_constrain_base
        # assert sorted(constrain_check_list) == list(range(curr_constrain_base))

        for alpha_idx in range(self.alpha_num):
            self.constrain_idx_h_sum[alpha_idx] = curr_constrain_base + alpha_idx
            constrain_check_list.append(self.constrain_idx_h_sum[alpha_idx])
            self.constrain_idx_beta_sum[alpha_idx] = curr_constrain_base + alpha_idx + self.alpha_num
            constrain_check_list.append(self.constrain_idx_beta_sum[alpha_idx])
        curr_constrain_base += self.alpha_num * 2
        # assert len(constrain_check_list) == curr_constrain_base
        # assert sorted(constrain_check_list) == list(range(curr_constrain_base))

        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                self.constrain_idx_beta[stage][alpha_idx] = curr_constrain_base + stage * self.alpha_num + alpha_idx
                constrain_check_list.append(self.constrain_idx_beta[stage][alpha_idx])
        curr_constrain_base += self.stage * self.alpha_num
        # assert len(constrain_check_list) == curr_constrain_base
        # assert sorted(constrain_check_list) == list(range(curr_constrain_base))

        for alpha_idx in range(self.alpha_num):
            base_for_alpha = curr_constrain_base + alpha_idx * self.exec_topk * 2 * n
            for i in range(self.exec_topk):
                base_for_i = base_for_alpha + i * n * 2
                self.constrain_idx_long_turnover_sum[alpha_idx][i] = base_for_i
                constrain_check_list += list(range(self.constrain_idx_long_turnover_sum[alpha_idx][i], self.constrain_idx_long_turnover_sum[alpha_idx][i] + n))
                self.constrain_idx_short_turnover_sum[alpha_idx][i] = base_for_i + n
                constrain_check_list += list(range(self.constrain_idx_short_turnover_sum[alpha_idx][i], self.constrain_idx_short_turnover_sum[alpha_idx][i] + n))
        curr_constrain_base += self.alpha_num * self.exec_topk * 2 * n
        assert len(constrain_check_list) == curr_constrain_base
        assert sorted(constrain_check_list) == list(range(curr_constrain_base))

        # Variables
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                for i in range(n):
                    task.putvarbound(self.val_idx_new_h[stage][alpha_idx] + i, mosek.boundkey.ra, -self.max_stage_inst_exposure, self.max_stage_inst_exposure)
                    task.putvarbound(self.val_idx_h_bound[stage][alpha_idx] + i, mosek.boundkey.ra, 0, self.max_stage_inst_exposure)
        aij = []

        for stage in range(self.stage):
            # Constraints: -h_bound <= new_h <= h_bound
            for alpha_idx in range(self.alpha_num):
                i = self.constrain_idx_h_bound_1[stage][alpha_idx]
                for j in range(n):
                    aij.extend([(i + j, self.val_idx_new_h[stage][alpha_idx] + j, 1), (i + j, self.val_idx_h_bound[stage][alpha_idx] + j, 1)])
                    task.putconbound(i + j, mosek.boundkey.lo, 0, float("inf"))
                i = self.constrain_idx_h_bound_2[stage][alpha_idx]
                for j in range(n):
                    aij.extend([(i + j, self.val_idx_new_h[stage][alpha_idx] + j, 1), (i + j, self.val_idx_h_bound[stage][alpha_idx] + j, -1)])
                    task.putconbound(i + j, mosek.boundkey.up, -float("inf"), 0)

            # Constraints: -turnover <= new_h - h(t) <= turnover
            for alpha_idx in range(self.alpha_num):
                i = self.constrain_idx_turnover[stage][alpha_idx]
                for j in range(n):
                    if alpha_idx == 0:
                        aij += [(i + j, self.val_idx_new_h[stage][alpha_idx] + j, 1)]
                        aij += [(i + j, self.val_idx_long_turnover[stage][alpha_idx][k] + j, -1) for k in range(self.exec_topk)]
                        aij += [(i + j, self.val_idx_short_turnover[stage][alpha_idx][k] + j, 1) for k in range(self.exec_topk)]
                    else:
                        aij += [(i + j, self.val_idx_new_h[stage][alpha_idx] + j, 1)]
                        aij += [(i + j, self.val_idx_new_h[stage][alpha_idx - 1] + j, -1)]
                        aij += [(i + j, self.val_idx_long_turnover[stage][alpha_idx][k] + j, -1) for k in range(self.exec_topk)]
                        aij += [(i + j, self.val_idx_short_turnover[stage][alpha_idx][k] + j, 1) for k in range(self.exec_topk)]

        # Constraints: sum(h_bound) <= 1
        for alpha_idx in range(self.alpha_num):
            i = self.constrain_idx_h_sum[alpha_idx]
            for stage in range(self.stage):
                for j in range(n):
                    aij.append((i, self.val_idx_h_bound[stage][alpha_idx] + j, 1))
            task.putconbound(i, mosek.boundkey.ra, 0, 1)

        for alpha_idx in range(self.alpha_num):
            i = self.constrain_idx_beta_sum[alpha_idx]
            for stage in range(self.stage):
                for j in range(n):
                    aij.append((i, self.val_idx_new_h[stage][alpha_idx] + j, 1))
            task.putconbound(i, mosek.boundkey.ra, -self.max_beta_exposure_sum, self.max_beta_exposure_sum)

        # Constraints: sum(stage_turnover) <= turnover_limit
        for alpha_idx in range(self.alpha_num):
            for k in range(self.exec_topk):
                i = self.constrain_idx_long_turnover_sum[alpha_idx][k]
                for stage in range(self.stage):
                    for j in range(n):
                        aij.append((i + j, self.val_idx_long_turnover[stage][alpha_idx][k] + j, 1))

                i = self.constrain_idx_short_turnover_sum[alpha_idx][k]
                for stage in range(self.stage):
                    for j in range(n):
                        aij.append((i + j, self.val_idx_short_turnover[stage][alpha_idx][k] + j, 1))

        task.putaijlist(*zip(*aij))
        return task

    def update_one_line(
        self,
        task,
        last_row,
        alphas,
        # probs,
        std,
        valid,
        fee,
        short_price,
        short_vol,
        long_price,
        long_vol,
        turnover_ma0,
        turnover_ma1,
        # turnover_ma2,
        corr1,
        corr2,
        corr3,
        oi,
        funding_fee,
        ret1m,
        # market_ret1m,
        book1_value_sum0,
        book1_value_sum1,
        timestamp,
    ):
        # adjust ep holding by ret1m
        last_row = [l * (1 + ret1m) for l in last_row]
        holding_abs_sum = sum([h.abs().sum() for h in last_row])
        if holding_abs_sum > 1:
            last_row = [h / holding_abs_sum for h in last_row]
        if not valid.any():
            return last_row

        n = len(last_row[0])
        universe = last_row[0].index
        # 生成用于划分 group 的索引
        # insts_values = (-pd.Series(std, index=universe)).sort_values()
        # group_idx = insts_values * 0
        # group_idx = pd.Series(0, index=insts_values.index, dtype=int)
        # group_size = n // self.group_num
        # for i in range(self.group_num):
        #     if i != self.group_num - 1:
        #         group_idx[insts_values.index[i * group_size:(i + 1) * group_size]] = i
        #     else:
        #         group_idx[insts_values.index[i * group_size:]] = i
        # self.group_idx = group_idx.loc[universe].values

        alphas = np.array(alphas)
        # probs = np.array(probs)
        if oi is None:
            oi = np.array([np.inf for _ in range(n)])
        if funding_fee is None:
            funding_fee = np.array([0 for _ in range(n)])

        mid_price = (short_price[0] + long_price[0]) / 2
        # for i, inst in enumerate(last_row[0].index):
        #     if mid_price[i] == 0 or not np.isfinite(mid_price[i]):
        #         print(timestamp, inst, "mid price is 0 or nan")
        long_cost = np.array([(long_price[k] / mid_price - 1) + fee + self.req_margin for k in range(self.exec_topk)])
        short_cost = np.array([-(short_price[k] / mid_price - 1) + fee + self.req_margin for k in range(self.exec_topk)])
        adjust_hold_risk = fee + (long_price[0] - short_price[0]) / mid_price / 2

        clip_funding_fee = np.sign(funding_fee) * np.clip(np.abs(funding_fee) - self.funding_fee_thres, 0, np.inf)
        long_cost = np.clip(long_cost + clip_funding_fee * self.funding_fee_coef, 0, np.inf)
        short_cost = np.clip(short_cost + clip_funding_fee * self.funding_fee_coef, 0, np.inf)

        # Objective
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                for i in range(n):
                    if valid[i]:
                        # if alpha_idx == 0:
                        #     task.putcj(self.val_idx_new_h[stage][alpha_idx] + i, alphas[alpha_idx][i] * self.signal_coef[self.signal_group_idx[i]][stage][alpha_idx])
                        # else:
                        #     task.putcj(
                        #         self.val_idx_new_h[stage][alpha_idx] + i, alphas[alpha_idx][i] * self.signal_coef[self.signal_group_idx[i]][stage][alpha_idx] -
                        #         alphas[alpha_idx - 1][i] * self.signal_coef[self.signal_group_idx[i]][stage][alpha_idx - 1])
                        task.putcj(self.val_idx_new_h[stage][alpha_idx] + i, alphas[alpha_idx][i] * self.signal_coef[self.signal_group_idx[i]][stage][alpha_idx])
                        task.putcj(
                            self.val_idx_h_bound[stage][alpha_idx] + i,
                            -(std[i] * std[i] * self.inst_risk_coef[stage][alpha_idx]) - adjust_hold_risk[i] * self.hold_risk_coef[self.cost_group_idx[i]][stage][alpha_idx],
                        )
                        for k in range(self.exec_topk):
                            task.putcj(
                                self.val_idx_long_turnover[stage][alpha_idx][k] + i,
                                (-long_cost[k][i]) * self.turnover_coef[self.cost_group_idx[i]][stage][alpha_idx],
                            )
                            task.putcj(
                                self.val_idx_short_turnover[stage][alpha_idx][k] + i,
                                (-short_cost[k][i]) * self.turnover_coef[self.cost_group_idx[i]][stage][alpha_idx],
                            )
                    else:
                        task.putcj(self.val_idx_new_h[stage][alpha_idx] + i, 0)
                        task.putcj(self.val_idx_h_bound[stage][alpha_idx] + i, 0)
                        for k in range(self.exec_topk):
                            task.putcj(self.val_idx_long_turnover[stage][alpha_idx][k] + i, 0)
                            task.putcj(self.val_idx_short_turnover[stage][alpha_idx][k] + i, 0)
        # Variables
        long_limit = np.array([long_price[k] * long_vol[k] for k in range(self.exec_topk)])
        short_limit = np.array([short_price[k] * short_vol[k] for k in range(self.exec_topk)])
        # turnover_last_slow_ratio = turnover_ma0 / turnover_ma2
        # turnover_quick_slow_ratio = turnover_ma1 / turnover_ma2
        # long_probs = (probs + 1) / 2
        # short_probs = 1 - long_probs
        # org_probs = (probs + 1) / 2
        # org_probs = np.log(org_probs / (1 - org_probs))
        # long_probs = probs
        # short_probs = -probs
        # turnover_last_slow_ratio = (turnover_ma0.sum() / turnover_ma1.sum()) if (turnover_ma1 is not None) else None
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                for i in range(n):
                    if valid[i]:
                        # if corr1[i] < self.abnormal_corr_ban_open_thres[stage][0] or corr2[i] < self.abnormal_corr_ban_open_thres[stage][1] or corr3[i] < self.abnormal_corr_ban_open_thres[stage][
                        #         2] or abs(last_row[stage].values[i]) * self.nav > oi[i] * self.stage_oi_exposure:
                        #     # 相关性过低，异动太大，禁止开仓
                        #     # oi exposure 太大，禁止开仓
                        #     for k in range(self.exec_topk):
                        #         task.putvarbound(
                        #             self.val_idx_long_turnover[stage][alpha_idx][k] + i,
                        #             mosek.boundkey.ra,
                        #             0,
                        #             min(
                        #                 long_limit[k][i] * self.signal_horizon[alpha_idx] * self.turnover_limit_rate / self.nav,
                        #                 self.max_turnover[stage] * np.clip(oi[i] * self.stage_oi_exposure / self.nav / self.max_stage_inst_exposure, 0, 1),
                        #             ),
                        #         )
                        #         task.putvarbound(
                        #             self.val_idx_short_turnover[stage][alpha_idx][k] + i,
                        #             mosek.boundkey.ra,
                        #             0,
                        #             min(
                        #                 short_limit[k][i] * self.signal_horizon[alpha_idx] * self.turnover_limit_rate / self.nav,
                        #                 self.max_turnover[stage] * np.clip(oi[i] * self.stage_oi_exposure / self.nav / self.max_stage_inst_exposure, 0, 1),
                        #             ),
                        #         )
                        #     max_stage_inst_exposure = abs(last_row[stage].values[i])
                        #     task.putvarbound(self.val_idx_new_h[stage][alpha_idx] + i, mosek.boundkey.ra, -max_stage_inst_exposure, max_stage_inst_exposure)
                        #     task.putvarbound(self.val_idx_h_bound[stage][alpha_idx] + i, mosek.boundkey.ra, 0, max_stage_inst_exposure)
                        #     continue
                        # long_turnover_coef = 1 / (1 + np.exp(-probs[alpha_idx][i] / self.prob_thres[stage][alpha_idx]))
                        # short_turnover_coef = 1 / (1 + np.exp(probs[alpha_idx][i] / self.prob_thres[stage][alpha_idx]))
                        # if last_row[stage].values[i] > 0:
                        #     long_turnover_coef = 1 / (1 + np.exp(-probs[alpha_idx][i] / self.open_prob_thres[stage][alpha_idx]))
                        #     short_turnover_coef = 1 / (1 + np.exp(probs[alpha_idx][i] / self.close_prob_thres[stage][alpha_idx]))
                        # elif last_row[stage].values[i] < 0:
                        #     long_turnover_coef = 1 / (1 + np.exp(-probs[alpha_idx][i] / self.close_prob_thres[stage][alpha_idx]))
                        #     short_turnover_coef = 1 / (1 + np.exp(probs[alpha_idx][i] / self.open_prob_thres[stage][alpha_idx]))
                        # else:
                        #     long_turnover_coef = 1 / (1 + np.exp(-probs[alpha_idx][i] / self.open_prob_thres[stage][alpha_idx]))
                        #     short_turnover_coef = 1 / (1 + np.exp(probs[alpha_idx][i] / self.open_prob_thres[stage][alpha_idx]))
                        # TODO：把 ret1m 的符号当做 to 的符号，考虑一下带方向的 to
                        if EXCHANGE in ["okx10m", "binance5m", "okx5m"]:
                            oi_too_large_flag = abs(last_row[stage].values[i]) * self.nav > oi[i] * self.stage_oi_exposure
                            corr_too_low_flag = corr1[i] < self.abnormal_corr_ban_open_thres[stage][0] or corr2[i] < self.abnormal_corr_ban_open_thres[stage][1] or corr3[
                                i] < self.abnormal_corr_ban_open_thres[stage][2]
                            # liquidyty_too_low_flag = abs(last_row[stage].values[i]) > turnover_ma1[i] * 20 * self.turnover_limit_rate / self.nav / self.stage
                            # turnover_quick_too_large_flag = turnover_last_slow_ratio[i] > self.abnormal_turnver_too_large_quick_thres[stage]
                            # turnover_quick_too_small_flag = turnover_last_slow_ratio[i] < self.abnormal_turnver_too_small_quick_thres[stage]
                            # turnover_slow_too_large_flag = turnover_quick_slow_ratio[i] > self.abnormal_turnver_too_large_slow_thres[stage]
                            # turnover_slow_too_small_flag = turnover_quick_slow_ratio[i] < self.abnormal_turnver_too_small_slow_thres[stage]
                            # can_long = (self.abnormal_turnver_ban_long_thres1[stage] > turnover_last_slow_ratio[i]) and (self.abnormal_turnver_ban_long_thres2[stage] > turnover_quick_slow_ratio[i])
                            # can_short = (self.abnormal_turnver_ban_short_thres1[stage] > turnover_last_slow_ratio[i]) and (self.abnormal_turnver_ban_short_thres2[stage] > turnover_quick_slow_ratio[i])
                            # can_open = not oi_too_large_flag and not corr_too_low_flag
                            # can_open = not oi_too_large_flag and not corr_too_low_flag and not liquidyty_too_low_flag
                            can_open = not oi_too_large_flag and not corr_too_low_flag

                            # ban_long = turnover_last_slow_ratio > self.abnormal_turnver_ban_open_beta[alpha_idx] and market_ret1m < 0
                            # ban_short = turnover_last_slow_ratio > self.abnormal_turnver_ban_open_beta[alpha_idx] and market_ret1m > 0
                            # can_long = not ban_long
                            # can_short = not ban_short

                            can_long = True
                            can_short = True

                            # can_trade = not turnover_quick_too_large_flag and not turnover_quick_too_small_flag and not turnover_slow_too_large_flag and not turnover_slow_too_small_flag

                            # abnormal_turnver_can_long = (self.abnormal_turnver_ban_long_thres1[stage] > turnover_last_slow_ratio[i]) and (self.abnormal_turnver_ban_long_thres2[stage]
                            #                                                                                                               > turnover_quick_slow_ratio[i])
                            # abnormal_turnver_can_short = (self.abnormal_turnver_ban_short_thres1[stage] > turnover_last_slow_ratio[i]) and (self.abnormal_turnver_ban_short_thres2[stage]
                            #                                                                                                                 > turnover_quick_slow_ratio[i])
                            # can_long = abnormal_turnver_can_long
                            # can_short = abnormal_turnver_can_short
                            # abnormal_turnver_ban_long = (self.abnormal_turnver_ban_long_thres1[stage] < turnover_last_slow_ratio[i]) or (self.abnormal_turnver_ban_long_thres2[stage]
                            #                                                                                                              < turnover_quick_slow_ratio[i])
                            # abnormal_turnver_ban_short = (self.abnormal_turnver_ban_short_thres1[stage] < turnover_last_slow_ratio[i]) or (self.abnormal_turnver_ban_short_thres2[stage]
                            #                                                                                                                < turnover_quick_slow_ratio[i])
                            # abnormal_turnver_ban_long = abnormal_turnver_ban_long and last_row[stage].values[i] >= 0
                            # abnormal_turnver_ban_short = abnormal_turnver_ban_short and last_row[stage].values[i] <= 0
                            # abnormal_ret1m_can_long = (abs(ret1m[i]) / std[i] < self.abnormal_ret1m_ban_long_thres[stage]) or (ret1m[i] > 0)
                            # abnormal_ret1m_can_short = (abs(ret1m[i]) / std[i] < self.abnormal_ret1m_ban_short_thres[stage]) or (ret1m[i] < 0)
                            # can_long = float(abnormal_turnver_can_long and abnormal_ret1m_can_long)
                            # can_short = float(abnormal_turnver_can_short and abnormal_ret1m_can_short)

                            # can_long = float(abnormal_turnver_can_long or last_row[stage].values[i] < 0)
                            # can_short = float(abnormal_turnver_can_short or last_row[stage].values[i] > 0)
                            # can_open_long = abnormal_turnver_can_long and not oi_too_large_flag and not corr_too_low_flag
                            # can_open_short = abnormal_turnver_can_short and not oi_too_large_flag and not corr_too_low_flag
                            # can_long = float(not abnormal_turnver_ban_long)
                            # can_short = float(not abnormal_turnver_ban_short)
                        elif EXCHANGE in ["CF", "CF5m"]:
                            # can_open_long = True
                            # can_open_short = True
                            can_open = True
                            can_long = True
                            can_short = True
                        else:
                            raise ValueError("unknown exhcange")
                        # is_abnormal_corr = corr1[i] < self.abnormal_corr_ban_open_thres[stage][0] or corr2[i] < self.abnormal_corr_ban_open_thres[stage][1] or corr3[
                        #     i] < self.abnormal_corr_ban_open_thres[stage][2]
                        # if last_row[stage].values[i] >= 0 and is_abnormal_corr:
                        #     can_long = 0
                        # if last_row[stage].values[i] <= 0 and is_abnormal_corr:
                        #     can_short = 0
                        for k in range(self.exec_topk):
                            task.putvarbound(
                                self.val_idx_long_turnover[stage][alpha_idx][k] + i,
                                mosek.boundkey.ra,
                                0,
                                min(
                                    long_limit[k][i] * self.signal_horizon[alpha_idx] * self.turnover_limit_rate / self.nav,
                                    (self.max_open_turnover[stage] if last_row[stage].values[i] >= 0 else self.max_close_turnover[stage]) *
                                    np.clip(oi[i] * self.stage_oi_exposure / self.nav / self.max_stage_inst_exposure, 0, 1),
                                ) if can_long else EPS,
                            )
                            task.putvarbound(
                                self.val_idx_short_turnover[stage][alpha_idx][k] + i,
                                mosek.boundkey.ra,
                                0,
                                min(
                                    short_limit[k][i] * self.signal_horizon[alpha_idx] * self.turnover_limit_rate / self.nav,
                                    (self.max_open_turnover[stage] if last_row[stage].values[i] <= 0 else self.max_close_turnover[stage]) *
                                    np.clip(oi[i] * self.stage_oi_exposure / self.nav / self.max_stage_inst_exposure, 0, 1),
                                ) if can_short else EPS,
                            )
                        max_stage_inst_exposure = max(self.max_stage_inst_exposure, abs(last_row[stage].values[i]))
                        # adjust_probs = 1 / (1 + np.exp(-org_probs[alpha_idx][i] * self.open_prob_thres[stage][alpha_idx]))
                        # adjust_probs = adjust_probs * 2 - 1
                        # adjust_probs = np.clip(adjust_probs / self.close_prob_thres[stage][alpha_idx], -1, 1)
                        # if adjust_probs >= 0:
                        #     if can_open:
                        #         max_stage_inst_long_exposure = max(self.max_stage_inst_exposure * adjust_probs, abs(last_row[stage].values[i]))
                        #     else:
                        #         max_stage_inst_long_exposure = abs(last_row[stage].values[i])
                        #     max_stage_inst_short_exposure = abs(last_row[stage].values[i])
                        # else:
                        #     if can_open:
                        #         max_stage_inst_short_exposure = max(self.max_stage_inst_exposure * (-adjust_probs), abs(last_row[stage].values[i]))
                        #     else:
                        #         max_stage_inst_short_exposure = abs(last_row[stage].values[i])
                        #     max_stage_inst_long_exposure = abs(last_row[stage].values[i])
                        max_stage_inst_long_exposure = max_stage_inst_exposure if can_open else abs(last_row[stage].values[i])
                        max_stage_inst_short_exposure = max_stage_inst_exposure if can_open else abs(last_row[stage].values[i])
                        task.putvarbound(self.val_idx_new_h[stage][alpha_idx] + i, mosek.boundkey.ra, -max_stage_inst_short_exposure, max_stage_inst_long_exposure)
                        task.putvarbound(self.val_idx_h_bound[stage][alpha_idx] + i, mosek.boundkey.ra, 0, max_stage_inst_exposure)
                    else:
                        max_stage_inst_exposure = max(self.max_stage_inst_exposure, abs(last_row[stage].values[i]))
                        task.putvarbound(self.val_idx_long_turnover[stage][alpha_idx][k] + i, mosek.boundkey.fx, 0, 0)
                        task.putvarbound(self.val_idx_short_turnover[stage][alpha_idx][k] + i, mosek.boundkey.fx, 0, 0)
                        task.putvarbound(self.val_idx_new_h[stage][alpha_idx] + i, mosek.boundkey.ra, -max_stage_inst_exposure, max_stage_inst_exposure)
                        task.putvarbound(self.val_idx_h_bound[stage][alpha_idx] + i, mosek.boundkey.ra, 0, max_stage_inst_exposure)

        # Constraints: sum(stage_turnover) <= turnover_limit
        for alpha_idx in range(self.alpha_num):
            for k in range(self.exec_topk):
                j = self.constrain_idx_long_turnover_sum[alpha_idx][k]
                for i in range(n):
                    task.putconbound(i + j, mosek.boundkey.ra, 0, long_limit[k][i] * self.signal_horizon[alpha_idx] * self.turnover_limit_rate / self.nav)

                j = self.constrain_idx_short_turnover_sum[alpha_idx][k]
                for i in range(n):
                    task.putconbound(i + j, mosek.boundkey.ra, 0, short_limit[k][i] * self.signal_horizon[alpha_idx] * self.turnover_limit_rate / self.nav)

        # Constraints
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                if alpha_idx == 0:
                    for i in range(n):
                        task.putconbound(self.constrain_idx_turnover[stage][alpha_idx] + i, mosek.boundkey.fx, last_row[stage].values[i], last_row[stage].values[i])
                else:
                    for i in range(n):
                        task.putconbound(self.constrain_idx_turnover[stage][alpha_idx] + i, mosek.boundkey.fx, 0, 0)

        # 每个合约波动率不同，所以需要用多份低风险合约对冲高风险合约
        self.today_beta = np.ones(n)

        aij = []
        # Constraints: -max_beta_exposure <= sum_beta(new_h) <= max_beta_exposure
        for stage in range(self.stage):
            for alpha_idx in range(self.alpha_num):
                i = self.constrain_idx_beta[stage][alpha_idx]
                for j in range(n):
                    aij.append((i, self.val_idx_new_h[stage][alpha_idx] + j, self.today_beta[j]))
                    # aij.append((i, self.val_idx_h_bound[stage][alpha_idx] + j, self.today_beta[j]))
                curr_beta = (last_row[stage] * self.today_beta).sum()
                # curr_beta = (last_row[stage].abs() * self.today_beta).sum()
                beta_exposure = max(self.max_beta_exposure[stage], abs(curr_beta))
                task.putconbound(i, mosek.boundkey.ra, -beta_exposure, beta_exposure)

        turnover_last_slow_ratio = ((turnover_ma0.sum() / book1_value_sum0.sum()) / (turnover_ma1.sum() / book1_value_sum1.sum())) if (turnover_ma1 is not None) else None
        curr_beta = sum((last_row[stage] * self.today_beta).sum() for stage in range(self.stage))
        # if turnover_last_slow_ratio > self.abnormal_turnver_ban_open_beta:
        #     beta_constraint = max(self.max_beta_exposure_sum, abs(curr_beta))
        # else:
        #     beta_constraint = abs(curr_beta)
        for alpha_idx in range(self.alpha_num):
            i = self.constrain_idx_beta_sum[alpha_idx]
            if turnover_last_slow_ratio is not None and turnover_last_slow_ratio > self.abnormal_turnver_ban_open_beta[alpha_idx]:
                assert EXCHANGE in ["okx5m", "binance5m"]
                beta_constraint = abs(curr_beta)
            else:
                beta_constraint = max(self.max_beta_exposure_sum, abs(curr_beta))
            task.putconbound(i, mosek.boundkey.ra, -beta_constraint, beta_constraint)

        task.putaijlist(*zip(*aij))
        task.optimize()
        solsta = task.getsolsta(mosek.soltype.itr)
        # solsta = task.getsolsta(mosek.soltype.bas)    # TODO: 试试速度和回测
        if solsta == mosek.solsta.optimal:
            x = np.zeros(self.var_num)
            task.getxx(mosek.soltype.itr, x)
            line_list = [x[stage * self.stage_var_num:stage * self.stage_var_num + n] for stage in range(self.stage)]
            fix_line_list = [self.fix_line(pd.Series(line, index=universe)) for line in line_list]
            return fix_line_list
        else:
            print(f"solsta = {solsta} at {timestamp}")
            # embed()
            # sleep(1)
            # embed()
            return [self.fix_line(pd.Series(line, index=universe)) for line in last_row]

    def fix_line(self, line):
        # line = np.clip(line, -self.max_stage_inst_exposure, self.max_stage_inst_exposure)
        # TODO: fix beta exposure
        # TODO: fix risk to
        return line

    def update_one_day(
        self,
        date,
        last_row,
        alphas,
        # probs,
        std,
        fee,
        short_price,
        short_vol,
        long_price,
        long_vol,
        turnover_ma0,
        turnover_ma1,
        # turnover_ma2,
        corr1,
        corr2,
        corr3,
        oi,
        funding_fee,
        ret1m,
        # market_ret1m,
        book1_value_sum0,
        book1_value_sum1,
        valid,
    ):
        # task = self.build_task(alphas[0].columns, read_universe_values(date, self.univ_name))
        task = self.build_task(
            alphas[0].columns,
            # -pd.Series([read_alpha(date, inst, "std_r320").iloc[0] for inst in alphas[0].columns], index=alphas[0].columns),
            None,
            # read_universe_values(date, self.univ_name),
        )

        holding = pd.DataFrame(0, columns=alphas[0].columns, index=alphas[0].index, dtype=float)
        for i, ts in enumerate(alphas[0].index):
            # st = time.time()
            last_row = self.update_one_line(
                task=task,
                last_row=last_row,
                alphas=[alpha.values[i, :] for alpha in alphas],
                # probs=[prob.values[i, :] for prob in probs],
                std=std.values[i, :],
                valid=valid.values[i, :],
                fee=fee.values,
                short_price=[bp.values[i, :] for bp in short_price],
                short_vol=[bv.values[i, :] for bv in short_vol],
                long_price=[ap.values[i, :] for ap in long_price],
                long_vol=[av.values[i, :] for av in long_vol],
                turnover_ma0=turnover_ma0.values[i, :] if turnover_ma0 is not None else None,
                turnover_ma1=turnover_ma1.values[i, :] if turnover_ma1 is not None else None,
                # turnover_ma2=turnover_ma2.values[i, :],
                ret1m=ret1m.values[i, :],
                # market_ret1m=market_ret1m.values[i] if market_ret1m is not None else None,
                corr1=corr1.values[i, :] if corr1 is not None else None,
                corr2=corr2.values[i, :] if corr2 is not None else None,
                corr3=corr3.values[i, :] if corr3 is not None else None,
                oi=oi.values[i, :] if oi is not None else None,
                funding_fee=funding_fee.values[i, :] if funding_fee is not None else None,
                book1_value_sum0=book1_value_sum0.values[i, :],
                book1_value_sum1=book1_value_sum1.values[i, :],
                timestamp=ts,
            )
            holding.iloc[i] = sum(last_row)
            # print(f"cost {(time.time() - st)*1000:.2f} ms")
            # if abs((last_row * self.today_beta).sum()) > self.max_beta_exposure * 1.1:
            #     print((last_row * self.today_beta).sum())
        return holding, last_row

    def get_holding(self, dates):
        # prepare
        holdings = []
        pbar = tqdm(total=len(dates))
        last_row = [pd.Series(0.0, index=[]) for _ in range(self.stage)]

        for date in dates:
            try:
                # last_row = pd.Series(dtype='float64')
                insts = read_universe(date, self.univ_name).index
                today_index = get_index(date)
                alphas = [read_signal(date, self.univ_name, alpha_name) for alpha_name in self.alpha_name]
                # probs = [read_signal(date, self.univ_name, prob_name) for prob_name in self.prob_name]
                for alpha in alphas:
                    assert alpha.columns.equals(insts)
                    assert np.isfinite(alpha.values.flatten()).all()
                    # embed()
                    assert (alpha.index == today_index).all()
                # for prob in probs:
                #     assert prob.columns.equals(insts)
                #     assert np.isfinite(prob.values.flatten()).all()
                #     assert (prob.index == today_index).all()
                basedata = [read_basedata(date, inst) for inst in insts]
                book1_value_sum0 = pd.concat([read_alpha(date, inst, "book1_value_sum_1s_r1").rename(inst) for inst in insts], axis=1)
                book1_value_sum1 = pd.concat([read_alpha(date, inst, "book1_value_sum_1s_r200").rename(inst) for inst in insts], axis=1)
                adjust_book1_value_sum0 = book1_value_sum0 / self.turnover_limit_rate * self.book_limit_rate
                # adjust_book1_value_sum1 = book1_value_sum1 / self.turnover_limit_rate * self.book_limit_rate
                if self.exec_type == "take":
                    orderbook = [read_orderbook(date, inst) for inst in insts]
                    short_price = [
                        pd.concat([ob[f"bid{k}_price"].rename(inst).replace(0.0, np.nan).fillna(bd["close"]).to_frame()
                                   for inst, ob, bd in zip(insts, orderbook, basedata)], axis=1)
                        for k in range(1, self.exec_topk + 1)
                    ]
                    short_vol = [pd.concat([ob[f"bid{k}_volume"].rename(inst).to_frame() for inst, ob in zip(insts, orderbook)], axis=1) for k in range(1, self.exec_topk + 1)]
                    long_price = [
                        pd.concat([ob[f"ask{k}_price"].rename(inst).replace(0.0, np.nan).fillna(bd["close"]).to_frame()
                                   for inst, ob, bd in zip(insts, orderbook, basedata)], axis=1)
                        for k in range(1, self.exec_topk + 1)
                    ]
                    long_vol = [pd.concat([ob[f"ask{k}_volume"].rename(inst).to_frame() for inst, ob in zip(insts, orderbook)], axis=1) for k in range(1, self.exec_topk + 1)]
                elif self.exec_type == "take_bp":
                    orderbook = [read_orderbook(date, inst) for inst in insts]
                    short_price = [
                        pd.concat([ob[f"bid_vwap_{k}bp"].rename(inst).replace(0.0, np.nan).fillna(bd["close"]).to_frame()
                                   for inst, ob, bd in zip(insts, orderbook, basedata)], axis=1)
                        for k in range(1, self.exec_topk + 1)
                    ]
                    short_vol = [pd.concat([ob[f"bid_vol_{k}bp"].rename(inst).to_frame() for inst, ob in zip(insts, orderbook)], axis=1) for k in range(1, self.exec_topk + 1)]
                    long_price = [
                        pd.concat([ob[f"ask_vwap_{k}bp"].rename(inst).replace(0.0, np.nan).fillna(bd["close"]).to_frame()
                                   for inst, ob, bd in zip(insts, orderbook, basedata)], axis=1)
                        for k in range(1, self.exec_topk + 1)
                    ]
                    long_vol = [pd.concat([ob[f"ask_vol_{k}bp"].rename(inst).to_frame() for inst, ob in zip(insts, orderbook)], axis=1) for k in range(1, self.exec_topk + 1)]
                elif self.exec_type == "make":
                    short_price = [pd.concat([bd[f"close"].mul(1 - self.extra_slippage[k]).rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    short_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    short_turnover = [t.where(t < adjust_book1_value_sum0, adjust_book1_value_sum0) for t in short_turnover]
                    short_vol = [short_turnover[k] / short_price[k] / self.exec_topk for k in range(self.exec_topk)]
                    long_price = [pd.concat([bd[f"close"].mul(1 + self.extra_slippage[k]).rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    long_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    long_turnover = [t.where(t < adjust_book1_value_sum0, adjust_book1_value_sum0) for t in long_turnover]
                    long_vol = [long_turnover[k] / long_price[k] / self.exec_topk for k in range(self.exec_topk)]
                elif self.exec_type == "make2":
                    half_spread_mean = pd.concat([read_alpha(date, inst, "half_spread_mean").rename(inst) for inst in insts], axis=1).fillna(0)
                    short_price = [pd.concat([bd[f"close"].add(-half_spread_mean[inst]).rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    short_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    short_turnover = [t.where(t < adjust_book1_value_sum0, adjust_book1_value_sum0) for t in short_turnover]
                    short_vol = [short_turnover[k] / short_price[k] / self.exec_topk for k in range(self.exec_topk)]
                    long_price = [pd.concat([bd[f"close"].add(half_spread_mean[inst]).rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    long_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(insts, basedata)], axis=1) for k in range(self.exec_topk)]
                    long_turnover = [t.where(t < adjust_book1_value_sum0, adjust_book1_value_sum0) for t in long_turnover]
                    long_vol = [long_turnover[k] / long_price[k] / self.exec_topk for k in range(self.exec_topk)]
                else:
                    raise ValueError(f"exec_type {self.exec_type} not supported")
                # adjust_cost = read_adjust_cost(insts)
                # adjust_cost = adjust_cost.to_frame().T
                # adjust_cost.index = today_index[0:1]
                # adjust_cost = adjust_cost.reindex(today_index, method='ffill')
                # if EXCHANGE == "CF":
                #     base_cost = []
                #     for bd, inst in zip(basedata, insts):
                #         bid1_price = bd['bid1_price']
                #         ask1_price = bd['ask1_price']
                #         base_cost.append(((ask1_price - bid1_price) / ((ask1_price + bid1_price) / 2)).rename(inst).fillna(0))
                #     base_cost = pd.concat(base_cost, axis=1)
                #     fee = read_fee(date, insts)
                #     fee = fee.to_frame().T
                #     fee.index = today_index[0:1]
                #     fee = fee.reindex(today_index, method='ffill')
                #     base_cost = base_cost + fee
                # else:
                #     base_cost = adjust_cost * 0 + FEE
                last_row = [l.reindex(insts).fillna(0.0) for l in last_row]
                if EXCHANGE in ["CF", "CF10s", "CF5m"]:
                    fee = read_fee(date, insts)
                else:
                    fee = pd.Series(FEE, index=insts, dtype=float)
                if EXCHANGE in ["CF", "CF10s", "CF5m"]:
                    valid = pd.concat([read_alpha(date, inst, "valid").eq(1).rename(inst) for inst in insts], axis=1)
                # elif EXCHANGE in ["okx10m", "binance5m", "binance"]:
                # elif EXCHANGE in ["okx10m", "binance5m"]:
                #     pdate = date
                #     ban_insts = set()
                #     rank_thres = {
                #         "okx10m": 0.1,
                #         "binance5m": 0.1,
                #         "binance": 0.4,
                #     }[EXCHANGE]
                #     for _ in range(3):
                #         pdate = prev_date(pdate)
                #         prev_ret1m = pd.concat([read_alpha(pdate, inst, "ret1m").rename(inst) for inst in insts], axis=1)
                #         market_ret1m = prev_ret1m.mean(axis=1)
                #         inst_market_corr = prev_ret1m.corrwith(market_ret1m, axis=0).sort_values(ascending=True)
                #         ban_insts |= set(inst_market_corr[:int(len(inst_market_corr) * rank_thres)].index.tolist())

                #     valid = pd.DataFrame(True, columns=insts, index=today_index, dtype=bool)
                #     valid.loc[:, list(ban_insts)] = False
                #     for l in last_row:
                #         valid.loc[:, l.abs().ge(EPS)] = True
                else:
                    valid = pd.DataFrame(True, columns=insts, index=today_index, dtype=bool)  # TODO: valid 怎么读？

                std = pd.concat([read_alpha(date, inst, self.std_name).rename(inst) for inst in insts], axis=1)
                if EXCHANGE in ["okx10m", "binance5m", "binance", "okx5m"]:
                    # turnover_ma0 = pd.concat([read_alpha(date, inst, "turnover_r1").rename(inst) for inst in insts], axis=1)
                    # turnover_ma1 = pd.concat([read_alpha(date, inst, "turnover_r10").rename(inst) for inst in insts], axis=1)
                    # turnover_ma2 = pd.concat([read_alpha(date, inst, "turnover_r200").rename(inst) for inst in insts], axis=1)
                    # assert (turnover_ma2.values.flatten() >= 0).all()
                    # assert (turnover_ma0.values.flatten()[turnover_ma2.values.flatten() == 0] == 0).all()
                    # assert (turnover_ma1.values.flatten()[turnover_ma2.values.flatten() == 0] == 0).all()
                    # turnover_ma2 = turnover_ma2.replace(0.0, EPS)

                    turnover_ma0 = pd.concat([read_alpha(date, inst, "turnover_r1").rename(inst) for inst in insts], axis=1)
                    turnover_ma1 = pd.concat([read_alpha(date, inst, "turnover_r200").rename(inst) for inst in insts], axis=1)
                    assert (turnover_ma1.values.flatten() >= 0).all()
                    assert (turnover_ma0.values.flatten()[turnover_ma1.values.flatten() == 0] == 0).all()
                    turnover_ma1 = turnover_ma1.replace(0.0, EPS)

                    corr1 = pd.concat([read_alpha(date, inst, f"corr_market_{self.univ_name}_r10").rename(inst) for inst in insts], axis=1)
                    corr2 = pd.concat([read_alpha(date, inst, f"corr_market_{self.univ_name}_r40").rename(inst) for inst in insts], axis=1)
                    corr3 = pd.concat([read_alpha(date, inst, f"corr_market_{self.univ_name}_r160").rename(inst) for inst in insts], axis=1)
                    oi = pd.concat([bd["open_interest"].rename(inst) for inst, bd in zip(insts, basedata)], axis=1)
                    funding_fee = pd.concat([read_alpha(date, inst, "funding_1hr_r24h") for inst in insts], axis=1)
                    # market_name = f"market_{self.univ_name}"
                    # market_ret1m = read_alpha(date, market_name, "ret1m").rename(market_name)
                elif EXCHANGE in ["CF5m"]:
                    corr1 = None
                    corr2 = None
                    corr3 = None
                    oi = None
                    funding_fee = None
                    turnover_ma0 = None
                    turnover_ma1 = None
                    # market_ret1m = None
                else:
                    # turnover_ma0 = pd.DataFrame(0.0, columns=insts, index=today_index)
                    # turnover_ma1 = pd.DataFrame(0.0, columns=insts, index=today_index)
                    # turnover_ma2 = pd.DataFrame(1.0, columns=insts, index=today_index)
                    # raise ValueError(f"market corr not supported for exchange {EXCHANGE}")
                    raise ValueError(f"market oi not supported for exchange {EXCHANGE}")
                ret1m = pd.concat([read_alpha(date, inst, "ret1m").rename(inst) for inst in insts], axis=1)
                # embed()
                holding, last_row = self.update_one_day(
                    date=date,
                    last_row=last_row,
                    alphas=alphas,
                    # probs=probs,
                    std=std,
                    fee=fee,
                    short_price=short_price,
                    short_vol=short_vol,
                    long_price=long_price,
                    long_vol=long_vol,
                    turnover_ma0=turnover_ma0,
                    turnover_ma1=turnover_ma1,
                    # turnover_ma2=turnover_ma2,
                    corr1=corr1,
                    corr2=corr2,
                    corr3=corr3,
                    oi=oi,
                    funding_fee=funding_fee,
                    ret1m=ret1m,
                    # market_ret1m=market_ret1m,
                    book1_value_sum0=book1_value_sum0,
                    book1_value_sum1=book1_value_sum1,
                    valid=valid,
                )
                # holding.iloc[-1] *= 0
                # holding.iloc[-1] = holding.iloc[-1].where(~valid.iloc[-1], 0.0)
                # last_row = [l.where(~valid.iloc[-1], 0.0) for l in last_row]
                holdings.append(holding)
            except Exception as e:
                print(f"error at {date}", traceback.format_exc())
                print(e)
                # embed()
                exit(0)
            pbar.update()

        return holdings


def parse_args():
    parser = ArgumentParser(description='deep alpha to holding')
    parser.add_argument('config', help='config file path')
    parser.add_argument("-j", "--jobs", default=-1, type=int, help="allow N jobs at once")
    parser.add_argument("-i", "--interval", default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    pass
    # args = parse_args()
    # with open(args.config, "r") as f:
    #     config = json5.load(f)
    # if args.interval is not None:
    #     if args.interval == "os":
    #         if EXCHANGE in ["binance5m", "okx10m"]:
    #             dates = get_dates("2025-09-01", "2026-04-13")
    #             # dates = get_dates("2026-01-01", "2026-04-13")
    #         elif EXCHANGE in ["okx5m"]:
    #             dates = get_dates("2025-09-01", "2026-04-13")
    #             # dates = get_dates("2026-01-01", "2026-04-13")
    #         elif EXCHANGE in ["CF5m"]:
    #             dates = get_dates("2025-08-01", "2026-01-27")
    #     else:
    #         raise ValueError(f"interval {args.interval} not supported")
    # else:
    #     dates = get_dates(config["start_date"], config["end_date"])
    # univ_name = config["univ_name"]
    # FEE = config["fee"]
    # strategy = GenPortfolio(
    #     **config["strategy"]["params"],
    #     alpha_name=config["alpha_name"],
    #     prob_name=config["prob_name"],
    #     univ_name=univ_name,
    #     nav=config["nav"],
    #     std_name=config["std_name"],
    # )
    # if args.jobs == -1:
    #     args.jobs = (len(dates) - 1) // 31 + 1
    # print(f"optimize, {args.jobs} jobs")
    # if args.jobs > 1:
    #     chunk_len = (len(dates) - 1) // args.jobs + 1
    #     params = [dates[i:i + chunk_len] for i in range(0, len(dates), chunk_len)]
    #     with Pool(args.jobs) as pool:
    #         holding_chunks = pool.map(strategy.get_holding, params)
    #     holdings = [holding for holding_chunk in holding_chunks for holding in holding_chunk]
    # else:
    #     holdings = strategy.get_holding(dates)

    # def save_holding(date, holding):
    #     write_holding(date, univ_name, config["out_name"], holding)

    # with Pool(12) as p:
    #     p.starmap(save_holding, zip(dates, holdings))
