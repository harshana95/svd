import argparse
import functools
import gc
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

import diffusers
import einops
import numpy as np
import scipy
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import get_scheduler
from huggingface_hub import create_repo, hf_hub_download, upload_folder, delete_repo, repo_exists, whoami
from tqdm import tqdm

from dataset import create_dataset
from models.Restormer_with_psf import RestormerKConfig, RestormerKModel
from utils.common_utils import crop_arr, keep_last_checkpoints, initialize, log_image
from psf.svpsf import PSFSimulator
from utils.utils import ordered_yaml
from utils.loss import Loss

logger = get_logger(__name__)

# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir):
    i = len(weights) - 1
    saved = {}
    while len(weights) > 0:
        weights.pop()
        model = models[i]

        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        model.save_pretrained(os.path.join(output_dir, f"{class_name}_{saved[class_name]}"))

        i -= 1

def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        # m = importlib.import_module(model._get_name()) # load the module, will raise ImportError if module cannot be loaded
        c = getattr(sys.modules[__name__],
                    class_name)  # get the class, will raise AttributeError if class cannot be found

        # load diffusers style into model
        load_model = c.from_pretrained(os.path.join(input_dir, f"{class_name}_{saved[class_name]}"))
        model.load_state_dict(load_model.state_dict())
        del load_model


def log_validation(model, dataloader, args, accelerator, step):
    gc.collect()
    torch.cuda.empty_cache()
    idx = 0
    model.eval()
    for batch in tqdm(dataloader):
        lq = batch['lq']
        gt = batch['gt']
        with torch.no_grad():
            out = model(lq)
        lq = lq.cpu().numpy()
        gt = gt.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = [lq[i], gt[i], out[i]]
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(accelerator, image1, f'{idx}', step)  # image format (N,C,H,W)
    model.train()

def main(args):
    # ================================================================================================== 1. Initialize
    accelerator = initialize(args, logger)

    # =============================================================================================== 2. Load dataset
    # create train and validation dataloaders
    train_set = create_dataset(args.datasets.train)
    dataloader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.train.batch_size,
        num_workers=args.datasets.train.get('num_worker_per_gpu', 1),
    )
    val_set = create_dataset(args.datasets.val)
    test_dataloader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=1,
        num_workers=args.datasets.val.get('num_worker_per_gpu', 1),
    )
    psf_data = PSFSimulator.load_psfs(args.datasets.train.name, 'PSFs_with_basis.h5')
    basis_psfs = psf_data['basis_psfs'][:]  # [:] is used to load the whole array to memory
    basis_weights = psf_data['basis_weights'][:]
    
    # ============================================================================================== 3. Load models
    if args.network.type == "RestormerK":
        psfs_cropped = crop_arr(torch.from_numpy(basis_psfs[:, 0]).to(torch.float32), args.datasets.train.gt_size, args.datasets.train.gt_size)
        psfs_cropped = einops.rearrange(psfs_cropped, 'b c h w -> 1 (b c) h w')
        if args.datasets.train.resize is not None:
            psfs_cropped = torch.nn.functional.interpolate(psfs_cropped, (args.datasets.train.resize, args.datasets.train.resize))
        psfs_cropped = psfs_cropped.numpy().tolist()
        config = RestormerKConfig(**args.network, psfs=psfs_cropped)
        model = RestormerKModel(config)

    # ========================================================================================== 4. setup for training
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.train.optim.scale_lr:
        args.train.optim.learning_rate = (
                args.train.optim.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.optim.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            print(f"Parameter {k} is not optimized.")

    params_to_optimize = [{'params': optim_params}]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.train.optim.learning_rate,
        betas=(args.train.optim.adam_beta1, args.train.optim.adam_beta2),
        weight_decay=args.train.optim.adam_weight_decay,
        eps=args.train.optim.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.train.gradient_accumulation_steps)
    if args.train.max_train_steps is None:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    print(f"lr={args.train.optim.learning_rate},        betas={args.train.optim.adam_beta1, args.train.optim.adam_beta2},        weight_decay={args.train.optim.adam_weight_decay},        eps={args.train.optim.adam_epsilon}")
    print(f"num_warmup_steps={args.train.scheduler.lr_warmup_steps},        num_training_steps={args.train.max_train_steps},        num_cycles={args.train.scheduler.lr_num_cycles},        power={args.train.scheduler.lr_power}")
    lr_scheduler = get_scheduler(
        args.train.scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=args.train.scheduler.lr_warmup_steps,
        num_training_steps=args.train.max_train_steps,
        num_cycles=args.train.scheduler.lr_num_cycles,
        power=args.train.scheduler.lr_power,
    )
    criterion = Loss(args.train.loss).to(accelerator.device)

    model, optimizer, dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tensorboard cannot handle list types for config
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # ========================================================================================== 5. Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")
    
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.path.resume_from_checkpoint:
        if args.path.resume_from_checkpoint != "latest":
            path = os.path.basename(args.path.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.path.experiments_root)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.path.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.path.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.path.experiments_root, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, args.train.num_train_epochs+1):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                """
                Training step ========================================================================================= 
                """
                lq = batch['lq']
                gt = batch['gt']
                
                optimizer.zero_grad()
                pred = model(lq)

                losses = criterion(pred, gt)
                accelerator.backward(losses['all'])
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                optimizer.step()
                lr_scheduler.step()

                """
                END Training step ===================================================================================== 
                """
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)

                    if accelerator.is_main_process:
                        if global_step % args.train.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.train.checkpoints_total_limit is not None:
                                keep_last_checkpoints(args.path.experiments_root, args.train.checkpoints_total_limit, logger)

                            save_path = os.path.join(args.path.experiments_root, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if global_step % args.train.validation_steps == 0:
                            log_validation(model, test_dataloader,
                                args, accelerator, global_step,
                            )
                    global_step += 1

                logs = {
                    "loss": losses['all'].detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                    }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.train.max_train_steps:
                    break

    # END TRAINING
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_id = args.hub_model_id or Path(args.output_dir).name
            if repo_exists(whoami()['name'] + "/" + repo_id):
                logger.warning(f"Deleting repo... {repo_id}")
                delete_repo(whoami()['name'] + "/" + repo_id)
            repo_id = create_repo(
                repo_id=repo_id,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id
            # save_model_card(
            #     repo_id,
            #     image_logs=image_logs,
            #     base_model=args.pretrained_model_name_or_path,
            #     repo_folder=args.output_dir,
            # )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.path.experiments_root,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    main(opt)