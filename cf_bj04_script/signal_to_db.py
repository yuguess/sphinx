import os
import time
import pandas as pd
from argparse import ArgumentParser
from traceback import print_exc

from sphinx.util.cf_exchange_api import get_dates, read_universe
from sphinx.util.exchange_api import write_signal


def parse_args():
    parser = ArgumentParser(description="strategy-dl signal to db")
    parser.add_argument("test_dir", type=str, help="dir: test_dir_name")
    parser.add_argument("-k", "--pred_key", type=str, required=True, help="prediction key saved to db")
    parser.add_argument("-i", "--interval", type=str, required=True, help="start and end date, format: YYYY-MM-DD/YYYY-MM-DD")
    parser.add_argument("-u", "--universe", type=str, required=True)
    parser.add_argument("-c", "--channel", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    
    st, ed = args.interval.split("/")
    dates = get_dates(st, ed)
    pred_dir = "pred"
    versions = sorted(version for version in os.listdir(args.test_dir) if os.path.isdir(f"{args.test_dir}/{version}/{pred_dir}"))
    if not versions:
        raise ValueError(f"no version with {pred_dir!r} found in {args.test_dir}")
    print(f"fusion versions: {versions}")

    for date in dates:
        try:
            universe = read_universe(date, args.universe).index

            univ_pred = [pd.read_pickle(f"{args.test_dir}/{v}/{pred_dir}/{date}_{args.channel}.pkl")[universe] for v in versions]
            assert all(df.columns.equals(universe) for df in univ_pred)
            univ_prob = [pd.read_pickle(f"{args.test_dir}/{v}/{pred_dir}/{date}_{args.channel}_prob.pkl")[universe] for v in versions]
            assert all(df.columns.equals(universe) for df in univ_prob)
            
            fusion_univ_pred = sum(univ_pred).astype(float) / len(versions)
            fusion_univ_prob = sum(univ_prob).astype(float) / len(versions)
            assert fusion_univ_pred.isna().sum().sum() == 0
            assert fusion_univ_prob.isna().sum().sum() == 0

            write_signal(date, args.universe, args.pred_key, fusion_univ_pred)
            write_signal(date, args.universe, f"{args.pred_key}_prob", fusion_univ_prob)
            print(f"{date} done")
        except Exception:
            print_exc()
            print(f"{date} fail")
            time.sleep(1)


if __name__ == "__main__":
    main()
