import os
import sys
from multiprocessing import Pool

import json5
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sphinx.util.exchange_api import get_dates, prev_date, read_alpha, read_basedata, read_holding, read_orderbook, sample_per_date


EPS = 1e-5
    
def read_data(date, config):
    print(date)
    
    holding = read_holding(date, config["univ_name"], config["out_name"])
    today_universe = holding.columns
    
    if date > config["start_date"]:
        prev_holding = read_holding(prev_date(date), config["univ_name"], config["out_name"])
        last_row = prev_holding.iloc[-1:].reindex(columns=today_universe, fill_value=0)
    else:
        last_row = pd.DataFrame(0.0, columns=today_universe, index=holding.index[-1:] - pd.Timedelta("1d"))

    holding = pd.concat([last_row, holding], axis=0).fillna(0).round(6)
    turnover_limit_rate = config["strategy"]["params"]["exec_info"]["turnover_limit_rate"]
    exec_ty = config["strategy"]["params"]["exec_info"]["exec_type"]
    basedata = [read_basedata(date, inst) for inst in today_universe]
    
    if exec_ty == "take":
        ret1m = pd.concat([read_alpha(date, inst, "midret1m").rename(inst) for inst in today_universe], axis=1).fillna(0)
        exec_topk = config["strategy"]["params"]["exec_info"]["topk"]
        orderbook = [read_orderbook(date, inst) for inst in today_universe]
        short_price = [
            pd.concat([ob[f"bid{k}_price"].rename(inst).replace(0.0, np.nan).fillna(bd["close"])
                       for inst, ob, bd in zip(today_universe, orderbook, basedata)], axis=1)
            for k in range(1, exec_topk + 1)
        ]
        short_volume = [pd.concat([ob[f"bid{k}_volume"].rename(inst) for inst, ob in zip(today_universe, orderbook)], axis=1) for k in range(1, exec_topk + 1)]
        long_price = [
            pd.concat([ob[f"ask{k}_price"].rename(inst).replace(0.0, np.nan).fillna(bd["close"])
                       for inst, ob, bd in zip(today_universe, orderbook, basedata)], axis=1)
            for k in range(1, exec_topk + 1)
        ]
        long_volume = [pd.concat([ob[f"ask{k}_volume"].rename(inst) for inst, ob in zip(today_universe, orderbook)], axis=1) for k in range(1, exec_topk + 1)]
    elif exec_ty == "take_bp":
        ret1m = pd.concat([read_alpha(date, inst, "midret1m").rename(inst) for inst in today_universe], axis=1).fillna(0)
        exec_topk = config["strategy"]["params"]["exec_info"]["topk"]
        orderbook = [read_orderbook(date, inst) for inst in today_universe]
        short_price = [
            pd.concat([ob[f"bid_vwap_{k}bp"].rename(inst).replace(0.0, np.nan).fillna(bd["close"])
                       for inst, ob, bd in zip(today_universe, orderbook, basedata)], axis=1)
            for k in range(1, exec_topk + 1)
        ]
        short_volume = [pd.concat([ob[f"bid_vol_{k}bp"].rename(inst) for inst, ob in zip(today_universe, orderbook)], axis=1) for k in range(1, exec_topk + 1)]
        long_price = [
            pd.concat([ob[f"ask_vwap_{k}bp"].rename(inst).replace(0.0, np.nan).fillna(bd["close"])
                       for inst, ob, bd in zip(today_universe, orderbook, basedata)], axis=1)
            for k in range(1, exec_topk + 1)
        ]
        long_volume = [pd.concat([ob[f"ask_vol_{k}bp"].rename(inst) for inst, ob in zip(today_universe, orderbook)], axis=1) for k in range(1, exec_topk + 1)]
    elif exec_ty == "make":
        ret1m = pd.concat([read_alpha(date, inst, "ret1m").rename(inst) for inst in today_universe], axis=1).fillna(0)
        extra_slippage = config["strategy"]["params"]["exec_info"]["extra_slippage"]
        exec_topk = len(extra_slippage)
        short_price = [pd.concat([bd[f"close"].mul(1 - extra_slippage[k]).rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        short_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        short_volume = [short_turnover[k] / short_price[k] / exec_topk for k in range(exec_topk)]
        long_price = [pd.concat([bd[f"close"].mul(1 + extra_slippage[k]).rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        long_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        long_volume = [long_turnover[k] / long_price[k] / exec_topk for k in range(exec_topk)]
    elif exec_ty == "make2":
        ret1m = pd.concat([read_alpha(date, inst, "ret1m").rename(inst) for inst in today_universe], axis=1).fillna(0)
        half_spread_mean = pd.concat([read_alpha(date, inst, "half_spread_mean").rename(inst) for inst in today_universe], axis=1).shift(-1).fillna(0)
        exec_topk = 1
        short_price = [pd.concat([bd[f"close"].add(-half_spread_mean[inst]).rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        short_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        short_volume = [short_turnover[k] / short_price[k] / exec_topk for k in range(exec_topk)]
        long_price = [pd.concat([bd[f"close"].add(half_spread_mean[inst]).rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        long_turnover = [pd.concat([bd[f"turnover"].rename(inst).to_frame() for inst, bd in zip(today_universe, basedata)], axis=1) for k in range(exec_topk)]
        long_volume = [long_turnover[k] / long_price[k] / exec_topk for k in range(exec_topk)]
    else:
        raise ValueError(f"exec_ty {exec_ty} not supported")

    mid_price = (short_price[0] + long_price[0]) / 2
    short_limit = [short_price[k] * short_volume[k] / config["nav"] for k in range(exec_topk)]
    long_limit = [long_price[k] * long_volume[k] / config["nav"] for k in range(exec_topk)]
    pre_pnl = holding.shift(1).iloc[1:] * ret1m
    trade = holding.diff().iloc[1:]
    current_long_turnover = trade.where(trade > 0, 0).abs()
    current_short_turnover = trade.where(trade < 0, 0).abs()
    topk_long_cost_list = []
    topk_short_cost_list = []
    for i in range(exec_topk):
        topk_long_cost = (current_long_turnover.clip(lower=0, upper=long_limit[i] * turnover_limit_rate) * (long_price[i] / mid_price - 1))
        topk_short_cost = (current_short_turnover.clip(lower=0, upper=short_limit[i] * turnover_limit_rate) * (1 - short_price[i] / mid_price))
        topk_long_cost_list.append(topk_long_cost)
        topk_short_cost_list.append(topk_short_cost)
        current_long_turnover = (current_long_turnover - long_limit[i] * turnover_limit_rate).clip(lower=0)
        current_short_turnover = (current_short_turnover - short_limit[i] * turnover_limit_rate).clip(lower=0)

    # assert np.allclose(current_long_turnover.iloc[1:], 0, atol=EPS)  # TODO：第一行涉及隔夜平仓，以后处理
    # assert np.allclose(current_short_turnover.iloc[1:], 0, atol=EPS)  # TODO：第一行涉及隔夜平仓，以后处理
    assert holding.abs().sum(axis=1).max() <= 1 + EPS

    fee = config["fee"]
    turnover = trade.abs()
    fee_cost = turnover * fee

    # cost = sum(stage_cost_list) / config["nav"] + holding.diff().iloc[1:] * vwap_slippage

    cost = sum(topk_long_cost_list) + sum(topk_short_cost_list) + fee_cost
    if exec_ty == "make":
        vwap_slippage = pd.concat([read_alpha(date, inst, "vwap_slippage").rename(inst) for inst in today_universe], axis=1).shift(-1).fillna(0)
        cost += trade * vwap_slippage
    pnl = pre_pnl - cost

    # funding_fee = pd.concat([read_funding_fee(date, inst).rename(inst) for inst in today_universe], axis=1)
    # pnl -= funding_fee * holding.iloc[2:]
    
    # 第一行涉及并行，暂不检查
    # 误差来自 pct holding diff，所以在前面减去，否则 assert 可能失败
    # assert (((turnover - EPS * config["nav"]) / market_turnover).iloc[1:] <= 0.04).all().all()
    # assert holding.abs().sum(axis=1).max() <= 1 + EPS

    # print(date, (sum(stage_cost_list) / config["nav"] - holding.diff().iloc[1:] * vwap_slippage).sum().sum() / holding.diff().iloc[1:].abs().sum().sum() * 1e4)
    # u = [c[:-4] for c in pnl.columns]
    # pnl.columns = u
    # turnover.columns = u
    # cost.columns = u
    return pnl, turnover, cost  #, holding.iloc[1:]


def main():
    cfg_pth = sys.argv[1]
    with open(cfg_pth, "r") as f:
        config = json5.load(f)
    POOL_NUM = int(sys.argv[2])

    dates = get_dates(config["start_date"], config["end_date"])

    with Pool(POOL_NUM) as p:
        data = p.starmap(read_data, ((d, config) for d in dates))

    pnl_list = [d[0] for d in data]
    all_cost = sum([d[2].sum().sum() for d in data])
    # cost_list = [d[2] for d in data]
    # pd.Series(cost_list).plot()
    minute_pnl = [p.sum(axis=1) for p in pnl_list]
    minute_pnl = pd.concat(minute_pnl, axis=0)
    # p = pd.concat(pnl_list, axis=0)
    # p = p.loc[:, p.sum().abs().ge(EPS)]
    # p.cumsum().iloc[::1440].plot()
    # p.sum().sort_values()
    # p.sum().sort_values().iloc[-50:]
    turnover_list = [d[1] for d in data]
    # holding_list = [d[3] for d in data]
    # holding_list = [d[3] for d in data]
    # a = sum([h.abs().sum().sum() for h in holding_list])
    # b = sum([h.diff().abs().sum().sum() / 2 for h in holding_list])
    # a / b # 平均持仓时间，TODO：如果超过预测值，可能需要增加预测频段过增加 tol
    # h = pd.concat(holding_list, axis=0)
    daily_pnl = []
    daily_to = []
    for i, date in enumerate(dates):
        daily_pnl.append(pnl_list[i].sum().sum())
        daily_to.append(turnover_list[i].sum().sum())

    daily_to = pd.Series(daily_to, index=pd.to_datetime(dates))
    daily_pnl = pd.Series(daily_pnl, index=pd.to_datetime(dates))
    equity = daily_pnl.cumsum() + 1
    minute_equity = minute_pnl.cumsum() + 1

    equity.plot(label="equity").legend()
    minute_equity.plot(label="minue_equity").legend()
    # (daily_pnl.resample('W').sum().cumsum() + 1).reindex(equity.index).interpolate().fillna(1).plot(label="weekly_pnl").legend()
    # (daily_pnl.resample('M').sum().cumsum() + 1).reindex(equity.index).interpolate().fillna(1).plot(label="monthly_pnl").legend()
    daily_to.plot(secondary_y=True, label="turnover").legend(loc="center left")

    # mask_dates = ["2023-12-31", "2024-01-01"]
    # masked_daily_pnl = daily_pnl.copy()
    # for mask_date in mask_dates:
    #     if mask_date in masked_daily_pnl.index:
    #         masked_daily_pnl.loc[mask_date] = 0

    sharp = (daily_pnl.mean() - 0.05 / 365) / daily_pnl.std() * ((365)**0.5)

    # 周度 sharp
    weekly_returns = daily_pnl.resample('W').sum()
    average_weekly_return = weekly_returns.mean()
    std_dev_weekly = weekly_returns.std()
    
    risk_free_rate_weekly = 0.05 / 52  # 一年有 52 周
    weekly_sharp = (average_weekly_return - risk_free_rate_weekly) / std_dev_weekly * ((52)**0.5)

    # 月度 sharp
    monthly_returns = daily_pnl.resample('ME').sum()
    average_monthly_return = monthly_returns.mean()
    std_dev_monthly = monthly_returns.std()
    
    risk_free_rate_monthly = 0.05 / 12  # 一年有 12 个月
    monthly_sharp = (average_monthly_return - risk_free_rate_monthly) / std_dev_monthly * ((12)**0.5)
    
    # masked_sharp = (masked_daily_pnl.mean() - 0.05 / 365) / masked_daily_pnl.std() * ((365)**0.5)
    turnover = daily_to.mean()
    minute_max_drawdown = (minute_equity - minute_equity.cummax().rolling(sample_per_date(), min_periods=1).mean()).min()

    # 两天就能回来的回撤可以忽略掉
    max_drawdown = (equity.rolling(2).min() - equity.rolling(2).min().cummax()).min()
    # max_drawdown = (equity - equity.cummax()).min()

    # masked_equity = masked_daily_pnl.cumsum() + 1
    # masked_max_drawdown = (masked_equity - masked_equity.cummax()).min()

    margin = daily_pnl.sum() / daily_to.sum() * 1e4
    # mrate = holdings.abs().sum(axis=1).mean()

    df = pd.DataFrame({
        "sharp": [sharp],
        "weekly_sharp": [weekly_sharp],
        "monthly_sharp": [monthly_sharp],
        # "masked_sharp": [masked_sharp],
        "pnl": [daily_pnl.sum()],
        "calmar": [-daily_pnl.sum() / len(daily_pnl) * 365 / max_drawdown],
        "minute_calmar": [-daily_pnl.sum() / len(daily_pnl) * 365 / minute_max_drawdown],
        # "masked_calmar": [-masked_daily_pnl.sum() / len(masked_daily_pnl) * 365 / masked_max_drawdown],
        "turnover": [turnover],
        "max_drawdown": [max_drawdown],
        "minute_max_drawdown": [minute_max_drawdown],
        # "masked_max_drawdown": [masked_max_drawdown],
        "margin": [margin],
        "cost": [all_cost * 1e4 / daily_to.sum()],
        # "mrate": [mrate],
        "win_rate": [(daily_pnl > 1e-4).mean()],
        "weekly_win_rate": [(weekly_returns > 1e-4).mean()],
        "monthly_win_rate": [(monthly_returns > 1e-4).mean()],
    })

    df.index = [f"nav {int(config['nav']/1e4)}w USDT"]
    # print(EXCHANGE, config["univ_name"], config["out_name"])
        
    assert "json5" in sys.argv[1]
    csv_name = cfg_pth.replace(".json5", ".csv")
    df.to_csv(csv_name)
    plt.savefig(f"{cfg_pth.replace('.json5', '.png')}")
    print(df)


if __name__ == "__main__":
    main()
