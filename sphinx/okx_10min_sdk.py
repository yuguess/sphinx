


def create_infra_sdk(trade_dt_s, cfg):
    sdk_wrapper = SDKWrapper(date=trade_dt_s, accounts=[cfg['account'][0]['name']], net_mode=False, univ_name=cfg["universe"])
    return sdk_wrapper
