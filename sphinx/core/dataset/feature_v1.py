import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool


def read_data(preifx, inst, start_date, end_date, label_name, sub_label_name, alphas):
    st_idx = get_index(start_date)[0]
    ed_idx = get_index(end_date)[-1]
    labels = [pd.read_pickle(f"{preifx}/{inst}/label/{l}.pkl.zip") for l in label_name]
    all_index = labels[0].index
    for l in labels:
        assert l.index.equals(all_index)
    valid_index = (labels[0].index >= st_idx) & (labels[0].index <= ed_idx)
    labels = [l.loc[valid_index].astype(np.float32) for l in labels]
    vola_name = [f"vola{''.join([i for i in l if i.isdigit()])}m" for l in label_name]
    # vola_name = [f"vola30m" for l in label_name]
    if len(labels[0]) > 0:
        if sub_label_name is not None:
            # if len(labels) == 1:
            #     label_sub = pd.read_pickle(f"{preifx}/{inst}/label/{sub_label_name}.pkl.zip")
            #     assert label_sub.index.equals(all_index)
            #     label_sub = label_sub.loc[valid_index].astype(np.float32)
            #     # label_sub = label_sub.iloc[valid_st_idx:valid_ed_idx].astype(np.float32)
            #     labels = [l - label_sub.values for l in labels]
            # else:
            assert sub_label_name == "prev"
            # assert len(labels) == 2
            for i in range(len(label_name) - 1, 0, -1):
                labels[i] = labels[i] - labels[i - 1].values
                # labels.append(labels[i] - labels[i - 1].values)
                # vola_name.append(vola_name[i])
        volas = []
        for l in vola_name:
            volas.append(pd.read_pickle(f"{preifx}/{inst}/label/{l}.pkl.zip").loc[valid_index].astype(np.float32))
        features = []
        for alpha in alphas:
            features.append(pd.read_pickle(f"{preifx}/{inst}/feature/{alpha}.pkl.zip").loc[valid_index].astype(np.float32))
            # features.append(pd.read_pickle(f"{preifx}/{inst}/feature/{alpha}.pkl.zip").iloc[valid_st_idx:valid_ed_idx].astype(np.float32))
        features = pd.concat(features, axis=1)
        labels = pd.concat(labels, axis=1)
        volas = pd.concat(volas, axis=1)
        return inst, labels, volas, features
    return None


class StdNorm:

    def __init__(self, df, clip):
        self.std = df.clip(lower=df.quantile(0.001, axis=0), upper=df.quantile(0.999, axis=0), axis=1).std(axis=0).values
        self.clip = clip

    def norm(self, df):
        df = df / self.std
        df = df.clip(-self.clip, self.clip)
        return df

    def div_std(self, df, dim):
        if dim is None:
            assert len(self.std) == 1
            return df / self.std[0]
        assert df.shape[dim] == len(self.std)
        shape = np.ones(len(df.shape), dtype=int)
        shape[dim] = len(self.std)
        std = self.std.reshape(shape)
        if isinstance(df, torch.Tensor):
            std = torch.tensor(std, device=df.device)
        return df / std

    def denorm(self, df, dim):
        if dim is None:
            assert len(self.std) == 1
            return df * self.std[0]
        assert df.shape[dim] == len(self.std)
        shape = np.ones(len(df.shape), dtype=int)
        shape[dim] = len(self.std)
        std = self.std.reshape(shape)
        if isinstance(df, torch.Tensor):
            std = torch.tensor(std, device=df.device)
        return df * std


if EXCHANGE == "coinbase":
    VOLA_THRES = 5e-6
else:
    VOLA_THRES = 5e-5


def merge_data(inst_name, features, org_labels, volas, use_vola, sample_per_date, date_symbols, all_time_index):
    features = pd.concat(features, axis=0)
    org_labels = pd.concat(org_labels, axis=0)
    volas = pd.concat(volas, axis=0).fillna(0)
    assert features.index.equals(org_labels.index) and features.index.equals(volas.index)
    assert features.index.is_monotonic_increasing
    assert features.index.is_unique
    # assert len(org_labels.index) % sample_per_date == 0
    will_use_index = all_time_index.values[date_symbols.values].flatten()
    features = features.reindex(will_use_index)
    org_labels = org_labels.reindex(will_use_index)
    volas = volas.reindex(will_use_index)

    vola_valid = volas.ge(VOLA_THRES)
    label_valid = org_labels.notna()
    if not use_vola:
        volas.loc[:] = np.expm1(1)
        volas = volas.where(vola_valid, 0)
    labels = np.sign(org_labels) * np.log1p(org_labels.abs()) / np.log1p(volas.values)
    labels = labels.where(vola_valid.values, 0).where(label_valid.values)
    org_labels = org_labels.where(vola_valid.values, 0).where(label_valid.values)
    return inst_name, features, org_labels, labels, volas


class FeatureBase(Dataset):
    def __init__(
        self,
        universe,
        label_name,
        alphas,
        start_date,
        end_date,
        seq_len,
        jump_step,
        downsample,
        valid_thr,
        mode,
        sub_label_name=None,
        use_vola=True,
        sample_weight_decay_coef=1,
        normalizer=None,
        flip_prob=0,
        insts=None,
        inst_list_only=False,
        # exchange_aug=False,
    ):
        super().__init__()
        assert alphas[0] == "ret1m"
        dates = get_dates(start_date, end_date)
        start_date = dates[0]
        end_date = dates[-1]

        # ret_list = []
        # exchange_list = ["okx", "binance"] if exchange_aug else [EXCHANGE]
        # for e in exchange_list:
        #     path_prefix = f"{data_root_dir(e)}/deep/{e}/{universe}"
        #     if insts is None:
        #         task_insts = [i for i in os.listdir(path_prefix)]
        #     else:
        #         task_insts = insts
        #     if len(task_insts) == 1:
        #         ret = [read_data(path_prefix, task_insts[0], start_date, end_date, label_name, sub_label_name, alphas)]
        #     else:
        #         with Pool(POOL_NUM) as p:
        #             ret = p.starmap(read_data, [(path_prefix, inst, start_date, end_date, label_name, sub_label_name, alphas) for inst in task_insts])
        #             ret = [x for x in ret if x is not None]
        #     ret_list.extend(ret)
        # for inst, label, vola, feature in ret_list:
        #     labels[inst] = label
        #     volas[inst] = vola
        #     features[inst] = feature
        path_prefix = f"{data_root_dir()}/deep/{EXCHANGE}/{universe}"
        if insts is None:
            insts = [i for i in os.listdir(path_prefix)]
        if len(insts) == 1:
            ret = [read_data(path_prefix, insts[0], start_date, end_date, label_name, sub_label_name, alphas)]
        else:
            with Pool(POOL_NUM) as p:
                ret = p.starmap(read_data, [(path_prefix, inst, start_date, end_date, label_name, sub_label_name, alphas) for inst in insts])
                ret = [x for x in ret if x is not None]
        
        labels = {}
        volas = {}
        features = {}
        for inst, label, vola, feature in ret:
            labels[inst] = label
            volas[inst] = vola
            features[inst] = feature
        self.insts = sorted(labels.keys())

        self.seq_len = seq_len
        self.sample_per_date_val = sample_per_date()
        # assert seq_len < self.sample_per_date_val  # get item 往后取一天，避免数据不足
        self.dates = dates
        self.universe = [read_universe(date, universe) for date in dates]
        self.universe_len = len(self.universe[-1])
        for ui, u in enumerate(self.universe):
            if len(u) != self.universe_len:
                print(f"[WARN] universe not equal: {dates[ui]}, {len(u)} vs {self.universe_len}")
        date_symbols = pd.concat([pd.Series(1.0, u.index) for u in self.universe], axis=1).T.fillna(0)
        date_symbols.index = pd.DatetimeIndex(dates)
        tmp = date_symbols.copy()
        date_symbols = date_symbols.astype(bool)
        if EXCHANGE == "CF":
            # 期货一天的数据少，多补一天
            date_symbols = date_symbols | tmp.shift(-2).fillna(0).astype(bool)
        date_symbols = date_symbols | tmp.shift(-1).fillna(0).astype(bool)
        date_symbols = date_symbols | tmp.shift(1).fillna(0).astype(bool)
        # if inst_list_only:
        #     return
        self.alphas = alphas
        valid = np.zeros((len(dates) * self.sample_per_date_val,), dtype=bool)
        if jump_step > 1:
            assert downsample == 1, f"downsample should be 1 instead of {downsample} when jump_step > 1"
        else:
            assert jump_step == 1
        valid[::jump_step] = True
        # 本该是 seq_len - 1，但是为了保证 get item 取到的是 valid 的数据，所以多减一个
        valid[-seq_len:] = False
        all_time_index = pd.concat([get_index(date).to_series().reset_index(drop=True) for date in dates], axis=1).T
        self.valids = pd.Series(valid, index=all_time_index.values.flatten())
        self.valid_idxs = np.where(self.valids)[0]
        self.sample_weight = self.valids[self.valids].index.year - pd.to_datetime(start_date).year
        self.sample_weight = [1 / sample_weight_decay_coef**k for k in self.sample_weight]

        self.features = {}
        self.org_labels = {}
        self.volas = {}
        self.labels = {}

        # 保证 insts 有序
        for inst in self.insts:
            inst_name = inst.split("_")[0]
            if inst_name not in date_symbols.columns:
                continue
            if inst_name not in self.org_labels:
                self.org_labels[inst_name] = []
                self.volas[inst_name] = []
                self.features[inst_name] = []
            self.org_labels[inst_name].append(labels[inst])
            self.volas[inst_name].append(volas[inst])
            self.features[inst_name].append(features[inst])

        self.inst_names = list(self.org_labels.keys())
        with Pool(POOL_NUM) as p:
            ret = p.starmap(merge_data, [(
                inst_name,
                self.features[inst_name],
                self.org_labels[inst_name],
                self.volas[inst_name],
                use_vola,
                self.sample_per_date_val,
                date_symbols[inst_name],
                all_time_index,
            ) for inst_name in self.inst_names])
        for inst_name, features, org_labels, labels, volas in ret:
            self.features[inst_name] = features
            self.org_labels[inst_name] = org_labels
            self.labels[inst_name] = labels
            self.volas[inst_name] = volas
        if normalizer is None:
            self.normalizer = {
                # invalid 的点都是 nan，不参与计算
                "feature": StdNorm(pd.concat(self.features.values(), axis=0), clip=5),
                "label": StdNorm(pd.concat(self.labels.values(), axis=0), clip=5),
                "org_label": StdNorm(pd.concat(self.org_labels.values(), axis=0), clip=5),
            }
        else:
            self.normalizer = normalizer
        assert len(self.normalizer["feature"].std) == len(self.alphas)
        assert len(self.normalizer["label"].std) == len(label_name)
        assert len(self.normalizer["org_label"].std) == len(label_name)

        self.norm_labels = {}
        self.norm_org_labels = {}
        self.norm_features = {}
        for inst_name in self.inst_names:
            self.norm_labels[inst_name] = (self.normalizer["label"].norm(self.labels[inst_name]))
            self.norm_org_labels[inst_name] = (self.normalizer["org_label"].norm(self.org_labels[inst_name]))
            self.norm_features[inst_name] = (self.normalizer["feature"].norm(self.features[inst_name]).fillna(0))

        del self.features

        self.downsample = downsample
        assert downsample == 1  # 真实 downsample 由外部的 sampler 完成
        self.flip_prob = flip_prob
        self.shuffle_idx = -1
        self.mode = mode

    def __len__(self):
        return len(self.valid_idxs) // self.downsample

    def __get_item__(self, idx):
        # 不要乱改，会影响到 test.py
        org_idx = idx
        if self.downsample > 1:
            idx = min(idx * self.downsample + np.random.randint(self.downsample), len(self.valid_idxs) - 1)
        item_idx = self.valid_idxs[idx]
        date_idx_ed = min((item_idx + self.seq_len - 1) // self.sample_per_date_val, len(self.dates) - 1)
        # item_index = self.valids.index[item_idx:item_idx + self.seq_len]
        item_index_st = self.valids.index[item_idx]
        item_index_ed = self.valids.index[item_idx + self.seq_len - 1]
        universe = self.universe[date_idx_ed].index
        # item_index = self.norm_org_labels[universe[0]].loc[item_index_st:item_index_ed].index

        x = []
        y = []
        y_true = []
        org_y = []
        vola = []
        # 选出来的就应该是 univ，与 test.py 配合
        for inst in universe:
            x.append(self.norm_features[inst].loc[item_index_st:item_index_ed].fillna(0).values)
            y.append(self.norm_labels[inst].loc[item_index_st:item_index_ed].values)
            y_true.append(self.org_labels[inst].loc[item_index_st:item_index_ed].values)
            org_y.append(self.norm_org_labels[inst].loc[item_index_st:item_index_ed].values)
            vola.append(self.volas[inst].loc[item_index_st:item_index_ed].fillna(0).values)
            # assert len(x[-1]) == self.seq_len
            # assert len(y[-1]) == self.seq_len
            # assert len(y_true[-1]) == self.seq_len
            # assert len(org_y[-1]) == self.seq_len
            # assert len(vola[-1]) == self.seq_len

        x = torch.tensor(np.stack(x), dtype=torch.float32).permute(0, 2, 1)  # N, C, T
        y = torch.tensor(np.stack(y), dtype=torch.float32).permute(0, 2, 1)
        y_true = torch.tensor(np.stack(y_true), dtype=torch.float32).permute(0, 2, 1)
        org_y = torch.tensor(np.stack(org_y), dtype=torch.float32).permute(0, 2, 1)
        vola = torch.tensor(np.stack(vola), dtype=torch.float32).permute(0, 2, 1)

        # ret 一般不是 0，如果 0 和 nan 一共在截面占比超过 50%，说明这个截面很可能有问题
        # 如果较多截面有问题，则不该使用这个样本
        # 可能是交易所原始数据的问题

        if EXCHANGE == "CF":
            flag = y_true.nan_to_num(0).eq(0).float().mean(dim=0).ge(0.8).flatten().float().mean() > 0.8
        elif EXCHANGE == "coinbase":
            flag = False
        else:
            flag = y_true.nan_to_num(0).eq(0).float().mean(dim=0).ge(0.5).flatten().float().mean() > 0.5
        if flag:
            print(f"err date: {self.dates[date_idx_ed]}, idx: {org_idx}, err ratio: {y_true.nan_to_num(0).eq(0).float().mean(dim=0).flatten().float().mean()}, time: {item_index_st}")
            if self.mode == "train":
                return self.__getitem__((org_idx + self.sample_per_date_val) % len(self))

        # augmentation
        if self.flip_prob > 0 and self.flip_prob >= np.random.rand():
            x, y, y_true, org_y = -x, -y, -y_true, -org_y

        # 早年间的 univ 截面数量不足，重复采样补充
        if x.shape[0] != self.universe_len:
            assert self.mode == "train"
            repeat_count = (self.universe_len + x.size(0) - 1) // x.size(0)
            x = x.repeat((repeat_count,) + (1,) * (x.dim() - 1))[:self.universe_len, ...]
            y = y.repeat((repeat_count,) + (1,) * (y.dim() - 1))[:self.universe_len, ...]
            y_true = y_true.repeat((repeat_count,) + (1,) * (y_true.dim() - 1))[:self.universe_len, ...]
            org_y = org_y.repeat((repeat_count,) + (1,) * (org_y.dim() - 1))[:self.universe_len, ...]
            vola = vola.repeat((repeat_count,) + (1,) * (vola.dim() - 1))[:self.universe_len, ...]
        return x, y, vola, y_true, org_y  #, item_index
