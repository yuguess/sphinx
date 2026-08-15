import os
import sys
import subprocess
from argparse import ArgumentParser
from multiprocessing import Pool
import pathlib

def parse_args():
    parser = ArgumentParser(description='deep test')
    parser.add_argument('test_dir', type=str, help='dir: test_dir_name / version')
    parser.add_argument('-e', '--epoch', type=str, help='test on model saved at the specific epoch')
    parser.add_argument('-g', '--gpu', nargs='+', type=int, default=[0, 0, 0, 0], help='gpu id list')
    parser.add_argument('-i', '--interval', type=str, default=None, help='start and end date to override, format: YYYYMMDD/YYYYMMDD')
    parser.add_argument('-s', '--state_dict_key', type=str, default='ema_state_dict', help='state dict key, sometimes we want to use ema state as state dict')
    parser.add_argument('-t', '--test_dataset', type=str, default='val_dataset')
    parser.add_argument('-x', '--script_module', type=str)
    return parser.parse_args()

def work(task, args, strat_root: str) -> None:
    i, sub_dir = task
    sub_dir_path = os.path.join(args.test_dir, sub_dir)
    script_module = args.script_module

    subprocess.run(
        [
            sys.executable, "-m", f"{script_module}.test", sub_dir_path, "-e", str(args.epoch), "-i", str(args.interval),
            "-s", str(args.state_dict_key), "-t", str(args.test_dataset), "-g", str(i)
        ], cwd=strat_root, check=True
    )

def main():
    module_root = str(pathlib.Path(__file__).parent.parent)
    print(module_root)
    args = parse_args()
    sub_dirs = sorted(os.listdir(args.test_dir))
    assert len(sub_dirs) == len(args.gpu)
    tasks = [(args.gpu[i], d) for i, d in enumerate(sub_dirs)]
    
    with Pool(len(tasks)) as p:
        p.starmap(work, ((t, args, module_root) for t in tasks))


if __name__ == "__main__":
    main()
