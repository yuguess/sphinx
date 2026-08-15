import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import pathlib
import pprint
from tqdm import tqdm

from mona.common.logging import get_logger
from sphinx.util.exchange_api import get_dates, get_index, prev_date, read_universe, sample_per_date
from sphinx.core.model import gen_model
from sphinx.core.dataset import gen_dataset
from sphinx.run_helper import add_model_legacy_path


LG = get_logger("test")

def parse_args():
    parser = ArgumentParser(description='deep_crypto test')
    parser.add_argument('test_dir', type=str, help='dir: test_dir_name / version')
    parser.add_argument('-e', '--epoch', type=str, help='test on model saved at the specific epoch')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-i', '--interval', type=str, default=None, help='start and end date to override, format: YYYYMMDD/YYYYMMDD')
    parser.add_argument('-s', '--state_dict_key', type=str, default='ema_state_dict', help='state dict key, sometimes we want to use ema state as state dict')
    parser.add_argument('-t', '--test_dataset', type=str, default='val_dataset')
    parser.add_argument('-r', '--rescale', type=float, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    ckpt_path = os.path.join(args.test_dir, f"{args.epoch}.pth.tar")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path} does not exists."

    checkpoint = torch.load(ckpt_path, weights_only=False)
    config = checkpoint["config"]
    
    # LG.info(pprint.pformat(config))
    
    assert args.interval is not None
    if args.interval is not None:
        st, ed = args.interval.split('/')
        config[args.test_dataset]["start_date"] = st
        config[args.test_dataset]["end_date"] = ed
    
    LG.info(config[args.test_dataset])
    dates = get_dates(config[args.test_dataset]["start_date"], config[args.test_dataset]["end_date"])
    
    test_model = gen_model(**config["model"], seq_len=config["dataset"]["seq_len"]).cuda()
    test_model.load_state_dict(checkpoint[args.state_dict_key])
    test_model.eval()
    normalizer = checkpoint["normalizer"]

    output_dir = f"{args.test_dir}/pred"
    os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    st_load_day = dates[0]
    extra_load_date = 10
    for _ in range(extra_load_date):
        st_load_day = prev_date(st_load_day)
    ed_load_day = config[args.test_dataset]["end_date"]
    test_dataset = gen_dataset(
        **{**config["dataset"], **config[args.test_dataset], **{"start_date": st_load_day, "end_date": ed_load_day}}, jump_step=1, normalizer=normalizer)
    
    # for testing
    test_dataset.valid_idxs = np.arange(len(test_dataset.valids)) 
    receptive_field = test_model.receptive_field
    out_channels = test_model.out_channels

    # dataset 往前多了一天，刚好够用
    # 目前按天处理，满足下列假设才能继续 infer，每天三次 infer
    seq_len = config["dataset"]["seq_len"]
    sample_per_date_val = sample_per_date()
    assert receptive_field < seq_len  # infer 三次的第一次要求
    infer_per_day = (sample_per_date_val - 1) // (test_dataset.seq_len - receptive_field) + 1
    
    all_labels = []
    all_preds = []
    all_probs = []
    pbar = tqdm(total=len(dates))
    for date_idx, date in enumerate(dates):
        universe = read_universe(date, config[args.test_dataset]["universe"])
        today_index = get_index(date)
        preds = [pd.DataFrame(index=today_index, columns=universe.index, dtype=np.float64) for _ in range(out_channels)]
        probs = [pd.DataFrame(index=today_index, columns=universe.index, dtype=np.float64) for _ in range(out_channels)]
        label = [pd.DataFrame(index=today_index, columns=universe.index, dtype=np.float64) for _ in range(out_channels)]
        volas = [pd.DataFrame(index=today_index, columns=universe.index, dtype=np.float64) for _ in range(out_channels)]
        
        # 从尾部开始，因为 get item 的 columns 是根据 item seq 尾部的 univ 取的
        item_idx = (date_idx + extra_load_date + 1) * sample_per_date_val - seq_len
        for i in range(infer_per_day):
            x, y, vola, y_true, org_y, _ = test_dataset.__getitem__(item_idx)
            with torch.no_grad():
                out, out_prob = test_model(x[np.newaxis, :].cuda())
                out = out[..., receptive_field:].data.cpu().numpy().reshape(len(universe.index), out_channels, seq_len - receptive_field)
                out_prob = out_prob.sigmoid()[..., receptive_field:].data.cpu().numpy().reshape(len(universe.index), out_channels, seq_len - receptive_field)

            y_true = y_true[..., receptive_field:].numpy().reshape(len(universe.index), out_channels, seq_len - receptive_field)
            vola = vola[..., receptive_field:].numpy().reshape(len(universe.index), out_channels, seq_len - receptive_field)
            decode_out = test_dataset.decode(out, vola, 1) * args.rescale
            decode_out_prob = out_prob * 2 - 1

            fill_idx_ed = len(today_index) - i * (test_dataset.seq_len - receptive_field)
            fill_idx_st = max(0, fill_idx_ed - (seq_len - receptive_field))
            fill_len = fill_idx_ed - fill_idx_st

            for channel in range(out_channels):
                preds[channel].iloc[fill_idx_st:fill_idx_ed] = decode_out[:, channel, -fill_len:].T
                probs[channel].iloc[fill_idx_st:fill_idx_ed] = decode_out_prob[:, channel, -fill_len:].T
                label[channel].iloc[fill_idx_st:fill_idx_ed] = y_true[:, channel, -fill_len:].T
                volas[channel].iloc[fill_idx_st:fill_idx_ed] = vola[:, channel, -fill_len:].T

            item_idx -= test_dataset.seq_len - receptive_field

        all_labels.append(label)
        all_preds.append(preds)
        all_probs.append(probs)
        pbar.update(1) 

    for date, label, pred, prob in zip(dates, all_labels, all_preds, all_probs):
        for channel in range(out_channels):
            pred[channel].to_pickle(f"{output_dir}/{date}_{channel}.pkl")
            prob[channel].to_pickle(f"{output_dir}/{date}_{channel}_prob.pkl")
            

if __name__ == "__main__":
    dl_root = os.environ['STRATEGY_DL_ROOT']
    add_model_legacy_path(str(pathlib.Path(dl_root) / "sphinx"))
    main()
