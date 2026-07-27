import datetime
import math
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def corr(a, b):
    return ((a * b).mean() - a.mean() * b.mean()) / (a.std() * b.std())


def symlink_force(target, link_name):
    target = os.path.split(target)[-1]
    link_name = os.path.abspath(link_name)
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        os.remove(link_name)
        os.symlink(target, link_name)


def save_checkpoint(state, output_dir, filename=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if filename is None:
        filename = os.path.join(output_dir, f"{state['epoch']}.pth.tar")
    torch.save(state, filename)
    last_name = os.path.join(output_dir, 'last.pth.tar')
    symlink_force(filename, last_name)


def save_best_ckpt(state, output_dir):
    filename = os.path.join(output_dir, f"{state['epoch']}.pth.tar")
    best_name = os.path.join(output_dir, 'best.pth.tar')
    if os.path.exists(filename) and os.path.isfile(filename):
        symlink_force(filename, best_name)
        return
    elif not os.path.exists(filename) or not os.path.isfile(filename):
        save_checkpoint(state, output_dir, filename=best_name)
    else:
        raise ValueError(f"unexpected checkpoint file state: {filename}")


class Logger:

    def __init__(self, path, verbose=False, dummy=False):
        self.dummy = dummy
        if dummy:
            return
        self.__path = f"{path}/main.log"
        self.tensorboard = SummaryWriter(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.__file = open(self.__path, "a")
        self.__verbose = verbose

    def log(self, message):
        if self.dummy:
            return
        now = datetime.datetime.now()
        if self.__verbose:
            print(f"{now}\t{message}")
        self.__file.write(f"{now}\t{message}\n")
        self.flush()

    def flush(self):
        if self.dummy:
            return
        if self.__file is not None:
            self.__file.flush()

    def flush_tensorboard(self):
        if self.dummy:
            return
        self.tensorboard.flush()

    def close(self):
        if self.dummy:
            return
        if self.__file is not None:
            self.__file.close()
            self.__file = None

    def add_scalar(self, *args, **kwargs):
        if self.dummy:
            return
        self.tensorboard.add_scalar(*args, **kwargs)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
            return
        elif self.val is not None:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            raise ValueError("unexpected meter state")


# def get_decoded_corr(normalizer, preds, labels, label_scales):
#     label_scales = np.log1p(label_scales)
#     labels = np.expm1(normalizer.denorm(labels) * label_scales)
#     preds = np.expm1(normalizer.denorm(preds) * label_scales)
#     mask = ~(np.isnan(labels) | np.isnan(preds))
#     labels = labels[mask]
#     preds = preds[mask]
#     corr_score_exact = corr(labels, preds)  # np.corrcoef(labels, preds, dtype=np.float32)[0, 1]
#     return corr_score_exact, mask


def get_corr(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = ~(np.isnan(labels) | np.isnan(preds))
    labels = labels[mask]
    preds = preds[mask]
    corr_score_exact = corr(labels, preds)  # np.corrcoef(labels, preds, dtype=np.float32)[0, 1]
    return corr_score_exact, mask


def get_section_corr(x, y):
    invalid = np.isnan(y)
    valid = ~invalid
    valid_sum_1 = np.sum(valid, axis=1, keepdims=True)
    valid_sum_1 = np.where(valid_sum_1 == 0, 1, valid_sum_1)
    valid_sum_1_squeeze = valid_sum_1.squeeze(1)
    y = np.where(invalid, 0, y)
    x = np.where(invalid, 0, x)
    vx = x - np.sum(x, axis=1, keepdims=True) / valid_sum_1
    vy = y - np.sum(y, axis=1, keepdims=True) / valid_sum_1
    vy = np.where(invalid, 0, vy)
    vx = np.where(invalid, 0, vx)
    a = ((vx * vy).sum(axis=1) / valid_sum_1_squeeze).flatten()
    b = np.sqrt(((vx**2).sum(axis=1) / valid_sum_1_squeeze)).flatten()
    c = np.sqrt(((vy**2).sum(axis=1) / valid_sum_1_squeeze)).flatten()
    valid = (b != 0) & (c != 0) & (np.isfinite(a))
    if not valid.any():
        return (np.zeros_like(a) * a).mean()
    corr = a[valid] / (b[valid] * c[valid])
    return corr.mean()


def get_channel_corr(x, y):
    # dim 0 is channel
    invalid = np.isnan(y)
    valid = ~invalid
    valid_sum_1 = np.sum(valid, axis=0, keepdims=True)
    valid_sum_1 = np.where(valid_sum_1 == 0, 1, valid_sum_1)
    y = np.where(invalid, 0, y)
    x = np.where(invalid, 0, x)
    vx = x - np.sum(x, axis=0, keepdims=True) / valid_sum_1
    vy = y - np.sum(y, axis=0, keepdims=True) / valid_sum_1
    vy = np.where(invalid, 0, vy)
    vx = np.where(invalid, 0, vx)

    valid_flatten = valid.flatten()
    vx = vx.flatten()[valid_flatten]
    vy = vy.flatten()[valid_flatten]
    vx = vx / np.std(vx)
    vy = vy / np.std(vy)
    corr = (vx * vy).mean()
    return corr


def get_pnl(preds, labels):
    mask = ~(np.isnan(labels) | np.isnan(preds))
    labels = labels[mask]
    preds = preds[mask]
    thres = np.quantile(preds, [0.05, 0.95])
    position = np.zeros_like(preds)
    # TODO：h_t = f(h_{t-1}, x_t)，而不是直接用 x_t 卡阈值
    position[preds > thres[1]] = 1
    position[preds < thres[0]] = -1
    pnl = np.mean(position * labels)
    return pnl
