import os
import json5
import torch
import datetime
import logging
import pickle
from argparse import ArgumentParser

from core.model import gen_model
from core.dataset import gen_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def parse_args():
    parser = ArgumentParser(description='deep_crypto train')
    parser.add_argument('config', help='config file path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-v', '--version', type=str, default=None)
    parser.add_argument('-w', '--weight', type=str, default=None)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--shuffle', action="store_true")
    return parser.parse_args()


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
):
    pass


def main():
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if args.version is None else args.version
    with open(args.config, "r") as f:
        config = json5.load(f)

    enable_shuffle = args.shuffle
    if enable_shuffle:
        config["output_dir"] += "_shuffle"
    output_dir = os.path.join(config["output_dir"], str(version))
    output_dir = output_dir.split("/")

    assert output_dir[0] in ["output"]
    output_dir.insert(1, EXCHANGE)
    tmp_dataset_dir = "/".join(["tmp_dataset"] + output_dir[1:-1])
    output_dir = "/".join(output_dir)
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
        checkpoint_path = args.resume if args.resume else args.weight
        if os.path.isfile(checkpoint_path):
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
        else:
            logger.log(f"=> no checkpoint found at '{checkpoint_path}'")

    logger.log('## init dataset ##')
    for key in ["data_loader", "train_loader", "val_loader"]:
        if "batch_size" in config[key]:
            assert config[key]["batch_size"] % accelerator.num_processes == 0
            config[key]["batch_size"] //= accelerator.num_processes

    if not enable_shuffle:
        downsample = config["train_dataset"]["downsample"]
        config["train_dataset"]["downsample"] = 1
        if accelerator.is_main_process:
            if not os.path.exists(f"{tmp_dataset_dir}/train.ok"):
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
        train_sampler = CustomWeightedRandomSampler(train_dataset.sample_weight, len(train_dataset) // downsample, replacement=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            **{
                **config["data_loader"],
                **config["train_loader"],
            },
            sampler=train_sampler,
        )
        normalizer = train_dataset.normalizer
    else:
        config["train_epoch"] = len(config["dataset"]["alphas"]) + 1
        config["save_freq"] = 10000

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
            )
        else:
            val_loader.dataset.set_shuffle_idx(epoch)
            test_loader.dataset.set_shuffle_idx(epoch)

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
        is_best = val_corr < best_corr
        best_corr = min(val_corr, best_corr)

        if accelerator.is_main_process:
            if (epoch + 1) % config["save_freq"] == 0 or epoch + 1 == config["train_epoch"]:
                logger.log(f"save checkpoint for epoch {epoch}")
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'ema_state_dict': ema_hook.state_dict(),
                        'state_dict': model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                        'grad_dict': optimizer.optimizer.state_dict() if accelerator.num_processes > 1 else optimizer.state_dict(),
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
                        'state_dict': model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                        'grad_dict': optimizer.optimizer.state_dict() if accelerator.num_processes > 1 else optimizer.state_dict(),
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
