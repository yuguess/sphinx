import os
import subprocess
import sys
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

from sphinx.util.path_utils import STRATEGY_DL_ROOT

# STRATEGY_DL_ROOT = Path(__file__).resolve().parent
# REPO_ROOT = STRATEGY_DL_ROOT.parent


def parse_args():
    parser = ArgumentParser(description='strategy-dl batch test')
    parser.add_argument('test_dir', type=str, help='dir: test_dir_name / version')
    parser.add_argument('-e', '--epoch', type=str, help='test on model saved at the specific epoch')
    parser.add_argument('-g', '--gpu', nargs='+', type=int, default=[0, 0, 0, 0], help='gpu id list')
    parser.add_argument('-i', '--interval', type=str, default=None, help='start and end date to override, format: YYYYMMDD/YYYYMMDD')
    parser.add_argument('-s', '--state_dict_key', type=str, default='ema_state_dict', help='state dict key, sometimes we want to use ema state as state dict')
    parser.add_argument('-t', '--test_dataset', type=str, default='val_dataset')
    return parser.parse_args()


args = parse_args()
sub_dirs = sorted(os.listdir(args.test_dir))
assert len(sub_dirs) == len(args.gpu)
tasks = [(args.gpu[i], d) for i, d in enumerate(sub_dirs)]


def work(task):
    i, sub_dir = task
    sub_dir_path = os.path.join(args.test_dir, sub_dir)
    subprocess.run(
        [
            sys.executable,
            "-m"
            "script.test",
            # str(STRATEGY_DL_ROOT + "/script/test.py"),
            sub_dir_path,
            "-e",
            str(args.epoch),
            "-i",
            str(args.interval),
            "-s",
            str(args.state_dict_key),
            "-t",
            str(args.test_dataset),
            "-g",
            str(i),
        ],
        cwd=STRATEGY_DL_ROOT, check=True,
    )

with Pool(len(tasks)) as p:
    p.map(work, tasks)
