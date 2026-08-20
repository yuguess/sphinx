import datetime
import logging
import os
import pickle
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import json5
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from IPython import embed
from torch import optim

from sphinx.util.path_utils import data_namespace_parts, STRATEGY_DL_ROOT
from sphinx.core.dataset import gen_dataset
from sphinx.core.hook import EMAHook
from sphinx.core.model import gen_model
from sphinx.core.model import loss as loss_func
from sphinx.core.utils import AverageMeter, Logger, get_channel_corr, get_corr, get_pnl, get_section_corr, save_best_ckpt, save_checkpoint


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """WeightedRandomSampler that supports large sample pools."""

    def __iter__(self):
        weights = self.weights.detach().cpu().numpy()
        prob = weights / weights.sum()
        rand_tensor = np.random.choice(len(weights), size=self.num_samples, replace=self.replacement, p=prob)
        return iter(rand_tensor.tolist())


def run_epoch(
    data_loader,
    model,
    ema_hook,
    criterion,
    optimizer,
    epoch,
    epoch_ty,
    accelerator,
    receptive_field,
    out_chaneels,
    logger,
    metric_downsample=1,
):
    total_loss = AverageMeter()
    batch_loss = AverageMeter()
    metric_downsample = max(1, int(metric_downsample))

    log_interval = max(1, round(len(data_loader) // 10))

    start = time.time()
    preds = [[] for _ in range(out_chaneels)]
    # probs = [[] for _ in range(out_chaneels)]
    labels_true = [[] for _ in range(out_chaneels)]
    label_scales = [[] for _ in range(out_chaneels)]
    prev_data = []
    for i, batch_data in enumerate(data_loader):
        x, y, scale, y_true, org_y, weight = batch_data
        x = x.cuda(non_blocking=True)
        out, out_prob = model.forward(x, adj=None)
        # x, y, scale, y_true, org_y = batch_data
        # out, out_prob = model.forward(x.cuda())

        # B, N, 1, T
        y = y[..., receptive_field:]
        out = out[..., receptive_field:]
        out_prob = out_prob[..., receptive_field:]
        scale = scale[..., receptive_field:]
        y_true = y_true[..., receptive_field:]
        org_y = org_y[..., receptive_field:]
        weight = weight[..., receptive_field:]
        y = y.cuda(non_blocking=True)
        scale = scale.cuda(non_blocking=True)
        y_true = y_true.cuda(non_blocking=True)
        org_y = org_y.cuda(non_blocking=True)
        weight = weight.cuda(non_blocking=True)

        x_dict = {}
        y_dict = {}
        x_dict["vola_norm_pred"] = out
        y_dict["vola_norm_label"] = y
        y_dict["vola"] = scale
        y_dict["true_label"] = y_true
        x_dict["novola_nonorm_pred"] = data_loader.dataset.decode(x_dict["vola_norm_pred"], y_dict["vola"], 2)
        y_dict["novola_nonorm_label"] = data_loader.dataset.decode(y_dict["vola_norm_label"], y_dict["vola"], 2)
        x_dict["novola_norm_pred"] = data_loader.dataset.normalizer["org_label"].div_std(x_dict["novola_nonorm_pred"], 2)
        y_dict["novola_norm_label"] = org_y
        # x_dict["novola_norm_add_near_dim3_pred"] = (x_dict["novola_norm_pred"][:, :, :, 1:] + x_dict["novola_norm_pred"][:, :, :, :-1]) / 1
        # y_dict["novola_norm_add_near_dim3_label"] = (y_dict["novola_norm_label"][:, :, :, 1:] + y_dict["novola_norm_label"][:, :, :, :-1]) / 1
        # y_dict["novola_norm_label"] = data_loader.dataset.normalizer["org_label"].div_std(y_dict["novola_nonorm_label"], 2).clip(-5, 5)
        x_dict["tanh_novola_norm_pred"] = torch.tanh(x_dict["novola_norm_pred"])
        y_dict["tanh_novola_norm_label"] = torch.tanh(y_dict["novola_norm_label"])
        # x_dict["dmean_novola_norm_pred"] = x_dict["novola_norm_pred"] - x_dict["novola_norm_pred"].mean(dim=1, keepdim=True)
        # y_dict["dmean_novola_norm_label"] = y_dict["novola_norm_label"] - y_dict["novola_norm_label"].mean(dim=1, keepdim=True)

        # mask = y_dict["novola_norm_label"].isnan()
        # x_dict["idx_novola_norm_pred"] = x_dict["novola_norm_pred"].masked_fill(mask, 0.0).mean(dim=1, keepdim=True)
        # y_dict["idx_novola_norm_label"] = y_dict["novola_norm_label"].masked_fill(mask, 0.0).mean(dim=1, keepdim=True)
        # x_dict["idx_novola_norm_pred"] = x_dict["novola_norm_pred"].mean(dim=1, keepdim=True)
        # y_dict["idx_novola_norm_label"] = y_dict["novola_norm_label"].mean(dim=1, keepdim=True)

        x_dict["prob_pred"] = out_prob
        y_dict["prob_label"] = org_y
        x_dict["prob_vola_pred"] = out_prob
        y_dict["prob_vola_label"] = y
        x_dict["tanh_prob_pred"] = out_prob
        y_dict["tanh_prob_label"] = y_dict["tanh_novola_norm_label"]

        # y_dict["prob_label"] = y_dict["novola_norm_label"].clone()
        # x_dict["vola_prob_pred"] = out_prob
        # y_dict["vola_prob_label"] = y_dict["vola_norm_label"]

        loss = criterion(x_dict, y_dict, weight)
        prev_data.append(batch_data + [loss, x_dict, y_dict])
        prev_data = prev_data[-10:]
        # print(loss)
        if torch.isnan(loss):
            logger.log(f"NaN detected in loss")
            if epoch_ty == "train":
                if accelerator.is_main_process:
                    embed()
            elif epoch_ty in ["val", "test"]:
                loss = torch.tensor(0.0).cuda()
            else:
                raise ValueError(f"unknown epoch_ty: {epoch_ty}")
        if epoch_ty == "train":
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            ema_hook.step(epoch)

        # B, N, 1, T
        # x, y, scale, y_true, org_y = batch_data

        total_loss.update(loss, np.prod(y.shape))
        batch_loss.update(loss, np.prod(y.shape))

        for channel in range(out_chaneels) if i % metric_downsample == 0 else ():
            preds[channel].append(x_dict["vola_norm_pred"][:, :, channel:channel + 1, :].data.cpu().numpy().astype(np.float32, copy=False))
            # probs[channel].append(x_dict["prob_pred"][:, :, channel:channel + 1, :].data.cpu().numpy().astype(np.float32))
            labels_true[channel].append(y_dict["true_label"][:, :, channel:channel + 1, :].cpu().numpy().astype(np.float32, copy=False))
            label_scales[channel].append(y_dict["vola"][:, :, channel:channel + 1, :].cpu().numpy().astype(np.float32, copy=False))

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start
            # pseudo_turnover = (out / out.std()).diff(dim=-1).abs().mean(dim=-1).mean()
            logger.log(f"Epoch Step: {i + 1} Loss: {batch_loss.avg:.6f} "
                       #    f"Turnover: {pseudo_turnover:.6f} "
                       f"Samples per Sec: {log_interval * x.shape[0] / elapsed:.1f}")
            start = time.time()
            batch_loss.reset()

    # C, B, N, 1, T
    preds = np.stack([np.concatenate(p) for p in preds])
    # probs = np.stack([np.concatenate(p) for p in probs])
    labels_true = np.stack([np.concatenate(lt) for lt in labels_true])
    label_scales = np.stack([np.concatenate(ls) for ls in label_scales])
    decode_preds = data_loader.dataset.decode(preds, label_scales, dim=0)
    C, B, N, _, T = decode_preds.shape
    assert labels_true.shape == decode_preds.shape

    def calc_channel_metrics(channel):
        channel_preds = decode_preds[channel]
        channel_labels = labels_true[channel]
        sec_corr = get_section_corr(channel_preds, channel_labels)
        corr_score_decoded, mask = get_corr(channel_preds, channel_labels)
        flat_preds = channel_preds.flatten()
        flat_labels = channel_labels.flatten()
        turnover = abs(np.diff(flat_preds)).mean()
        pred_scale = np.abs(channel_preds).mean()
        valid = np.isfinite(flat_labels)
        std_deviation = flat_preds[valid].mean() / flat_preds[valid].std() - flat_labels[valid].mean() / flat_labels[valid].std()
        return channel, sec_corr, corr_score_decoded, turnover, pred_scale, std_deviation, mask.sum(), len(mask)

    with ThreadPoolExecutor(max_workers=out_chaneels) as executor:
        channel_metrics = list(executor.map(calc_channel_metrics, range(out_chaneels)))

    for channel, sec_corr, corr_score_decoded, turnover, pred_scale, std_deviation, mask_sum, mask_len in channel_metrics:

        reduce_sec_corr = accelerator.reduce(torch.tensor(sec_corr).cuda(), reduction="mean")
        reduce_corr_score_decoded = accelerator.reduce(torch.tensor(corr_score_decoded).cuda(), reduction="mean")
        reduce_pred_scale = accelerator.reduce(torch.tensor(pred_scale).cuda(), reduction="mean")
        reduce_std_deviation = accelerator.reduce(torch.tensor(std_deviation).cuda(), reduction="mean")
        reduce_turnover = accelerator.reduce(torch.tensor(turnover).cuda(), reduction="mean")
        reduce_mask = accelerator.reduce(torch.tensor(mask_sum).cuda(), reduction="sum")
        reduce_len = accelerator.reduce(torch.tensor(mask_len).cuda(), reduction="sum")
        # TODO：用什么指标评估 prob
        logger.log(f"Channel {channel} DecCorr Score:\t{reduce_corr_score_decoded:.6f}")
        logger.log(f"Channel {channel} SecCorr Score:\t{reduce_sec_corr:.6f}")
        logger.log(f"Channel {channel} StdDev:\t{reduce_std_deviation:.6f}")
        logger.log(f"Channel {channel} Scale:\t{reduce_pred_scale:.6f}")
        logger.log(f"Channel {channel} Turnover:\t{reduce_turnover:.6f}")
        logger.log(f"Channel {channel} DecCorr valid:\t{reduce_mask}/{reduce_len}; {reduce_mask*100/reduce_len:.2f}%")
        logger.add_scalar(f"DecCorr{channel}/{epoch_ty}", reduce_corr_score_decoded, epoch)
        logger.add_scalar(f"SecCorr{channel}/{epoch_ty}", reduce_sec_corr, epoch)
        logger.add_scalar(f"StdDev{channel}/{epoch_ty}", reduce_std_deviation, epoch)
        logger.add_scalar(f"Scale{channel}/{epoch_ty}", reduce_pred_scale, epoch)
        logger.add_scalar(f"Turnover{channel}/{epoch_ty}", reduce_turnover, epoch)

    reduce_total_loss = accelerator.reduce(total_loss.avg, reduction="mean")
    logger.log(f"Total Loss:\t{reduce_total_loss:.6f}")
    logger.add_scalar(f"TotalLoss/{epoch_ty}", reduce_total_loss, epoch)

    # if out_chaneels > 1:
    #     channel_corr = get_channel_corr(decode_preds, labels_true)
    #     reduce_channel_corr = accelerator.reduce(torch.tensor(channel_corr).cuda(), reduction="mean")
    #     logger.log(f"ChannelCorr Score:\t{reduce_channel_corr:.6f}")
    #     logger.add_scalar(f"ChannelCorr/{epoch_ty}", reduce_channel_corr, epoch)
    logger.flush_tensorboard()

    return -corr_score_decoded


def parse_args():
    parser = ArgumentParser(description='strategy-dl train')
    parser.add_argument('config', help='config file path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-v', '--version', type=str, default=None)
    parser.add_argument('-w', '--weight', type=str, default=None)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--shuffle', action="store_true")
    return parser.parse_args()


def main():
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # accelerator = Accelerator()

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    version = args.version
    if args.version is None:
        version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    elif args.version is not None:
        pass
    else:
        raise ValueError("unexpected version state")

    with open(args.config, "r") as f:
        config = json5.load(f)

    enable_shuffle = args.shuffle
    if enable_shuffle:
        config["output_dir"] += "_shuffle"

    output_dir = os.path.join(config["output_dir"], str(version))
    output_parts = output_dir.split("/")
    assert output_parts[0] in ["output"]
    output_parts[1:1] = data_namespace_parts()
    tmp_dataset_dir = Path(STRATEGY_DL_ROOT) / "tmp_dataset" / Path(*output_parts[1:-1])
    output_dir = STRATEGY_DL_ROOT / Path(*output_parts)
    tmp_dataset_dir = str(tmp_dataset_dir)
    output_dir = str(output_dir)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tmp_dataset_dir, exist_ok=True)
    args.verbose = True
    logger = Logger(output_dir, args.verbose, not accelerator.is_main_process)
    logger.log(json5.dumps(config, indent=4))

    logger.log('## init model ##')
    model = gen_model(**config["model"], seq_len=config["dataset"]["seq_len"]).cuda()
    receptive_field = model.receptive_field
    out_chaneels = model.out_channels
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.log(f'=> model parameters: {total_params:.2f}M')

    criterion = getattr(loss_func, config["loss"]["name"])(**config["loss"]["params"])
    optimizer = getattr(optim, config["optim"]["name"])(model.parameters(), **config["optim"]["params"])

    start_epoch = 0
    best_corr = float("inf")
    ema_states = None
    if args.weight or args.resume:
        checkpoint_path = args.weight
        if args.resume:
            checkpoint_path = args.resume
        elif not args.resume:
            pass
        else:
            raise ValueError(f"unexpected resume state: {args.resume}")
        checkpoint_exists = os.path.isfile(checkpoint_path)
        if checkpoint_exists:
            logger.log(f"=> loading checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
            info = model.load_state_dict(checkpoint['state_dict'], False)
            logger.log(info)
            ema_states = checkpoint.get('ema_state_dict', None)
            normalizer = checkpoint["normalizer"]
            if args.resume:
                optimizer.load_state_dict(checkpoint['grad_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_corr = checkpoint['best_corr']
            elif not args.resume:
                pass
            else:
                raise ValueError(f"unexpected resume state: {args.resume}")
        elif not checkpoint_exists:
            logger.log(f"=> no checkpoint found at '{checkpoint_path}'")
        else:
            raise ValueError(f"unexpected checkpoint state: {checkpoint_path}")

    logger.log('## init dataset ##')
    for key in ["data_loader", "train_loader", "val_loader"]:
        if "batch_size" in config[key]:
            assert config[key]["batch_size"] % accelerator.num_processes == 0
            config[key]["batch_size"] //= accelerator.num_processes

    if not enable_shuffle:
        downsample = config["train_dataset"]["downsample"]
        config["train_dataset"]["downsample"] = 1

        if accelerator.is_main_process:
            train_cache_ok = os.path.exists(f"{tmp_dataset_dir}/train.ok")
            if train_cache_ok:
                try:
                    with open(f"{tmp_dataset_dir}/train.pkl", "rb") as f:
                        cached_train_dataset = pickle.load(f)
                    expected_decay = config["train_dataset"].get("sample_weight_decay_coef", 1)
                    train_cache_ok = (
                        hasattr(cached_train_dataset, "sample_weight") and getattr(cached_train_dataset, "sample_weight_decay_coef", 1) == expected_decay)
                except Exception:
                    train_cache_ok = False

            if not train_cache_ok:
                train_dataset = gen_dataset(**{
                    **config["dataset"],
                    **config["train_dataset"],
                })
                with open(f"{tmp_dataset_dir}/train.pkl", "wb") as f:
                    pickle.dump(train_dataset, f)
                with open(f"{tmp_dataset_dir}/train.ok", "w") as f:
                    f.write("ok")

        accelerator.wait_for_everyone()
        with open(f"{tmp_dataset_dir}/train.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        # embed()
        train_sampler = CustomWeightedRandomSampler(train_dataset.sample_weight, max(1, len(train_dataset) // downsample), replacement=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            **{
                **config["data_loader"],
                **config["train_loader"],
            },
            sampler=train_sampler,
        )
        normalizer = train_dataset.normalizer
    elif enable_shuffle:
        config["train_epoch"] = len(config["dataset"]["alphas"]) + 1
        config["save_freq"] = 10000
    else:
        raise ValueError(f"unexpected shuffle state: {enable_shuffle}")

    if accelerator.is_main_process:
        if not os.path.exists(f"{tmp_dataset_dir}/val.ok"):
            val_dataset = gen_dataset(
                **{
                    **config["dataset"],
                    **config["val_dataset"],
                    "jump_step": (config["dataset"]["seq_len"] - receptive_field),
                },
                normalizer=normalizer,
            )
            with open(f"{tmp_dataset_dir}/val.pkl", "wb") as f:
                pickle.dump(val_dataset, f)
            with open(f"{tmp_dataset_dir}/val.ok", "w") as f:
                f.write("ok")

    accelerator.wait_for_everyone()
    with open(f"{tmp_dataset_dir}/val.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    val_loader = torch.utils.data.DataLoader(val_dataset, **{
        **config["data_loader"],
        **config["val_loader"],
    })

    if accelerator.is_main_process:
        if not os.path.exists(f"{tmp_dataset_dir}/test.ok"):
            test_dataset = gen_dataset(
                **{
                    **config["dataset"],
                    **config["test_dataset"],
                    "jump_step": (config["dataset"]["seq_len"] - receptive_field),
                },
                normalizer=normalizer,
            )
            with open(f"{tmp_dataset_dir}/test.pkl", "wb") as f:
                pickle.dump(test_dataset, f)
            with open(f"{tmp_dataset_dir}/test.ok", "w") as f:
                f.write("ok")
    accelerator.wait_for_everyone()
    with open(f"{tmp_dataset_dir}/test.pkl", "rb") as f:
        test_dataset = pickle.load(f)
    test_loader = torch.utils.data.DataLoader(test_dataset, **{
        **config["data_loader"],
        **config["val_loader"],
    })

    ema_hook = EMAHook(model, ema_states=ema_states, **config['ema'])
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)
    scheduler = optim.lr_scheduler.StepLR(optimizer, **config["optim"]["lr_scheduler"])
    for epoch in range(start_epoch, config["train_epoch"]):
        if not enable_shuffle:
            logger.log(f"Training {epoch}------lr {scheduler.get_last_lr()[0]:.6f}---------------------------")
            model.train()
            train_loss = run_epoch(
                train_loader,
                model,
                ema_hook,
                criterion,
                optimizer,
                epoch,
                "train",
                accelerator,
                receptive_field,
                out_chaneels,
                logger,
                metric_downsample=downsample,
            )
        elif enable_shuffle:
            val_loader.dataset.set_shuffle_idx(epoch)
            test_loader.dataset.set_shuffle_idx(epoch)
        else:
            raise ValueError(f"unexpected shuffle state: {enable_shuffle}")
        
        logger.log(f"Validation_{epoch}------------------------------------------")
        with torch.no_grad():
            model.eval()
            ema_hook.apply_ema()
            val_corr = run_epoch(
                val_loader,
                model,
                ema_hook,
                criterion,
                None,
                epoch,
                "val",
                accelerator,
                receptive_field,
                out_chaneels,
                logger,
            )
            test_corr = run_epoch(
                test_loader,
                model,
                ema_hook,
                criterion,
                None,
                epoch,
                "test",
                accelerator,
                receptive_field,
                out_chaneels,
                logger,
            )
            ema_hook.restore()
        is_best = val_corr < best_corr
        best_corr = min(val_corr, best_corr)

        if accelerator.is_main_process:
            save_due = (epoch + 1) % config["save_freq"] == 0 or epoch + 1 == config["train_epoch"]
            if save_due or is_best:
                state_dict = model.state_dict()
                grad_dict = optimizer.state_dict()
                if accelerator.num_processes > 1:
                    state_dict = model.module.state_dict()
                    grad_dict = optimizer.optimizer.state_dict()
                elif accelerator.num_processes == 1:
                    pass
                else:
                    raise ValueError(f"unexpected process count: {accelerator.num_processes}")
            if save_due:
                print(f"save checkpoint for epoch {epoch}")
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'ema_state_dict': ema_hook.state_dict(),
                        'state_dict': state_dict,
                        'grad_dict': grad_dict,
                        'best_corr': best_corr,
                        'is_best': is_best,
                        'config': config,
                        'normalizer': normalizer
                    },
                    output_dir=output_dir)

            if is_best:
                save_best_ckpt(
                    {
                        'epoch': epoch,
                        'ema_state_dict': ema_hook.state_dict(),
                        'state_dict': state_dict,
                        'grad_dict': grad_dict,
                        # 'state_dict': model.state_dict(),
                        # 'grad_dict': optimizer.state_dict(),
                        'best_corr': best_corr,
                        'is_best': is_best,
                        'config': config,
                        'normalizer': normalizer
                    },
                    output_dir=output_dir)

        scheduler.step()


if __name__ == '__main__':
    main()
