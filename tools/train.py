'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import argparse
import time
import os.path as osp
from copy import deepcopy

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torchvision

from pytorch3d.structures import Meshes

import _init_paths

from configs          import cfg, update_config
from nets             import Model
from dataset          import get_dataloader
from solver           import get_optimizer, adjust_learning_rate_step
from utils.utils      import AverageMeter, ProgressMeter
from utils.visualize  import *
from utils.checkpoint import save_checkpoint, load_checkpoint

from utils.utils import (
    set_seeds_cudnn,
    initialize_cuda,
    create_logger_directories,
    load_camera_intrinsics
)

torch.autograd.set_detect_anomaly(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training (default: 1)')

    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training (default: 0)')

    args = parser.parse_args()

    return args


def train(cfg):

    args = parse_args()
    update_config(cfg, args)

    # ******************************************************************************** #
    # ---------- Basic Setups
    # ******************************************************************************** #
    # Create directories to save outputs & logs
    logger, output_dir, log_dir = create_logger_directories(
        cfg, args.rank, phase='train', write_cfg_to_file=True
    )

    # Set all seeds & cudNN
    set_seeds_cudnn(cfg, seed=cfg.SEED)

    # GPU device
    device = initialize_cuda(cfg, args.rank)

    # Tensorboard
    if cfg.LOG_TENSORBOARD:
        tb_writer = SummaryWriter(log_dir)

    # ******************************************************************************** #
    # ---------- Build Model
    # ******************************************************************************** #
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    net    = Model(cfg, fov=camera['horizontalFOV'], device=device)

    # ******************************************************************************** #
    # ---------- Build Dataloaders
    # ******************************************************************************** #
    train_loader = get_dataloader(cfg, split='train')
    val_loader   = get_dataloader(cfg, split='validation')

    # ******************************************************************************** #
    # ---------- Build Optimizer & scaler
    # ******************************************************************************** #
    optimizer = get_optimizer(cfg, net)

    # Load checkpoint?
    checkpoint_file = osp.join(output_dir, f'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and osp.exists(checkpoint_file):
        last_epoch = load_checkpoint(
                        checkpoint_file,
                        net,
                        optimizer,
                        None,
                        device)
        begin_epoch = last_epoch
    else:
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        last_epoch  = -1

    # ******************************************************************************** #
    # ---------- Main Loop
    # ******************************************************************************** #
    # Freeze renderer (necessary?)
    for param in net.renderer.parameters():
        param.requires_grad = False

    # Main loops
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        batch_time = AverageMeter('', 'ms', ':3.0f')

        # --- Meters
        loss_train_meters = {}
        for l_name in net.loss_names:
            loss_train_meters[l_name] = AverageMeter(l_name, '', ':.2e')

        loss_val_meters = deepcopy(loss_train_meters)

        # --- Progress
        progress_train = ProgressMeter(
            len(train_loader),
            batch_time,
            list(loss_train_meters.values()),
            prefix="Epoch {:03d} ".format(epoch+1))

        progress_val = ProgressMeter(
            len(val_loader),
            batch_time,
            list(loss_val_meters.values()),
            prefix="Epoch {:03d} ".format(epoch+1))

        # -=-=-=-=-=- TRAINING LOOP -=-=-=-=-=- #
        net.train()
        for step, batch in enumerate(train_loader):

            # Adjust learning rate
            adjust_learning_rate_step(optimizer, epoch, step, len(train_loader), cfg)

            start = time.time()

            # ========== To CUDA
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ========== Forward pass
            optimizer.zero_grad(set_to_none=True)
            loss, sm = net(batch)

            # ========== Update
            loss.backward()
            clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            # ========== Elapsed time
            batch_time.update((time.time() - start) * 1000)

            # ========== Record loss
            for k, v in sm.items():
                loss_train_meters[k].update(float(v), cfg.TRAIN.BATCH_SIZE_PER_GPU)

            # Update logger for console
            progress_train.display(step+1, lr=optimizer.param_groups[0]['lr'])

        progress_train.display_summary()

        # -=-=-=-=-=- VALIDATION LOOP -=-=-=-=-=- #
        net.eval()
        if (epoch + 1) % cfg.TRAIN.VALID_FREQ == 0:
            for step, batch_val in enumerate(val_loader):

                # To CUDA
                batch_val = {k: v.to(device, non_blocking=True) for k, v in batch_val.items()}

                # Loss
                with torch.no_grad():
                    loss, sm = net(batch_val)

                # Record loss
                for k, v in sm.items():
                    loss_val_meters[k].update(float(v), cfg.TEST.BATCH_SIZE_PER_GPU)

                # Update logger for console
                progress_val.display(step+1, lr=optimizer.param_groups[0]['lr'])

            progress_val.display_summary()

        # ----- Update tensorboard
        if cfg.LOG_TENSORBOARD:
            for meter in loss_train_meters.values():
                tb_writer.add_scalar('Train/' + meter.name, meter.avg, epoch+1)

            if (epoch + 1) % cfg.TRAIN.VALID_FREQ == 0:
                for meter in loss_val_meters.values():
                    tb_writer.add_scalar('Validation/' + meter.name, meter.avg, epoch+1)

            # Input image
            imgs = []
            for b in range(4):
                imgs.append(denormalize(batch["image"][b]))

            imgs = torchvision.utils.make_grid(imgs, nrow=2)
            tb_writer.add_image('Images/Input Images', imgs, epoch+1)

            # ========== Inference
            with torch.no_grad():
                _, mesh, pcl, trans, rot = net.forward_encoder_generator(
                    batch["image"][:4], train=False
                )

                # Render (using predicted pose)
                R = torch.bmm(rot.transpose(1, 2), net.Rz.repeat(rot.shape[0], 1, 1))
                silh = net.renderer(mesh, R=R, T=trans)

                # Decompose mesh
                mesh_batch = []
                for b in range(4):
                    meshes = []
                    for i in range(cfg.MODEL.NUM_MAX_PRIMITIVES):
                        meshes.append(
                            Meshes(
                                verts=[pcl[b, :, i].detach().cpu()],
                                faces=[net.mesh_converter.faces.cpu()]
                            )
                        )

                    mesh_batch.append(meshes)

            # Plot mesh
            imgs = []
            tmp_fn = osp.join(log_dir, 'tmp.jpeg')
            for b in range(4):
                plot_3dmesh(mesh_batch[b], markers_for_vertices=False, savefn=tmp_fn)
                imgs.append(torchvision.io.read_image(tmp_fn))

            imgs = torchvision.utils.make_grid(imgs, nrow=2)
            tb_writer.add_image('Images/Reconstructed Assemblies', imgs, epoch+1)

            # Plot reprojection
            imgs = silh[:4, ..., 3].unsqueeze(1).mul(255).clamp(0,255).byte().cpu()
            imgs = torchvision.utils.make_grid(imgs, nrow=2)
            tb_writer.add_image('Images/Projection of Predictions', imgs, epoch+1)

            # GT masks
            imgs = torchvision.utils.make_grid(batch["mask"][:4].unsqueeze(1), nrow=2)
            tb_writer.add_image('Images/Ground-Truth Masks', imgs, epoch+1)

        # --- Save checkpoint
        if (epoch + 1) % cfg.TRAIN.VALID_FREQ == 0:
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'best_state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True, epoch+1 == cfg.TRAIN.END_EPOCH, output_dir)


if __name__ == "__main__":
    train(cfg)
