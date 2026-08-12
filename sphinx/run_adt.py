import os
import time
import numpy as np
import pandas as pd

from mona.common.logging import get_logger
from .okx_5min_sdk import create_infra_sdk, SDKWrapper, is_trading_time


LOGGER = get_logger('run_adt')


class LoopCtx:

    def __init__(self, cfg, parent_cfg_path, hist_row_csv_path, sdk_cli: SDKWrapper):
        self.account_name = cfg["name"]
        self.strategy_cfg = cfg["strategy"]
        self.stage = len(cfg["strategy"]["max_beta_exposure"])

        if hist_row_csv_path is None:
            self.hist_row = [pd.Series(np.nan) for _ in range(self.stage)]
        elif hist_row_csv_path == "deploy":
            self.hist_row = [sdk_cli.deploy_read_last_holding(self.account_name) / self.stage for _ in range(self.stage)]
        elif hist_row_csv_path == "last":
            last_path_prefix = f"deploy/data/{self.account_name}"
            last_path_dir = sorted(os.listdir(last_path_prefix))[-1]
            last_filename = sorted([i for i in os.listdir(os.path.join(last_path_prefix, last_path_dir)) if i.endswith(".csv")])[-1]
            last_file_path = os.path.join(last_path_prefix, last_path_dir, last_filename)
            print(f"[INFO] read last holding from {last_file_path}")
            df = pd.read_csv(last_file_path, index_col=0)
            self.hist_row = [df.loc[f"holding_stage{i}(%)"] / 100 for i in range(self.stage)]
            assert f"holding_stage{self.stage}(%)" not in df.index
        else:
            df = pd.read_csv(hist_row_csv_path, index_col=0)
            self.hist_row = [df.loc[f"holding_stage{i}(%)"] / 100 for i in range(self.stage)]
            assert f"holding_stage{self.stage}(%)" not in df.index

        self.deploy_last_row = pd.Series(np.nan)
        self.fusion_row = [pd.Series(np.nan) for _ in range(self.stage)]
        self.nav = -1
        self.run_name = ""
        self.parent_cfg_path = parent_cfg_path
        self.curr_date = ""
        self.valid_insts = pd.Series()

    def reset_holding(self):
        self.hist_row = [pd.Series(np.nan) for _ in range(self.stage)]
        self.deploy_last_row = pd.Series(np.nan)
        self.fusion_row = [pd.Series(np.nan) for _ in range(self.stage)]

    def try_reset(self, xhg, task_date, sdk_cli):
        curr_nav = sdk_cli.deploy_read_nav(self.account_name)

        if curr_nav != self.nav or self.curr_date != task_date:
            # if curr_nav != self.nav:
            #     self.reset_holding()
            self.curr_date = task_date
            self.nav = curr_nav
            self.run_name = f"{self.account_name}/{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
            os.makedirs(f"deploy/data/{self.run_name}", exist_ok=True)
            os.system(
                f'cp "{self.parent_cfg_path}" "deploy/data/{self.run_name}/config.json5"'
            )
            os.system(f'echo "{curr_nav}" > "deploy/data/{self.run_name}/nav.txt"')
            LOGGER.info("run_name: %s", self.run_name)