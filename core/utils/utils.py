'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import os
import sys
import numpy as np
import random
import logging
import time
import json
from pathlib  import Path
from enum     import Enum
from scipy.io import savemat

import torch
import torch.distributed as dist
# from torchinfo import summary

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# For reporting & logging
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """ Computes and stores the average and current value

        Modified from
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, unit='-', fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt  = fmt
        self.unit = unit
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def all_reduce(self, device):
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '' if not self.name else '{name} '
        fmtstr += '{val' + self.fmt + '}'
        if self.summary_type == Summary.AVERAGE:
            fmtstr += ' ({avg' + self.fmt + '})'
        elif self.summary_type == Summary.SUM:
            fmtstr += ' ({sum' + self.fmt + '})'
        elif self.summary_type == Summary.COUNT:
            fmtstr += ' ({count' + self.fmt + '})'

        fmtstr += ' {unit}'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('Invalid summary type %r' % self.summary_type)

        fmtstr += ' {unit}'

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """ Prints training progress

        Modified from
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, num_batches, timer, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.num_batches = num_batches
        self.timer  = timer
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, lr=None):
        entries = ['\r' + self.prefix + self.batch_fmtstr.format(batch)] # Epoch & Batch

        if lr is not None:
            entries += [f' (lr: {lr:.2e})']

        entries += ['[' + str(self.timer) + ']']
        entries += [str(meter) for meter in self.meters]
        msg = ' '.join(entries)

        if batch < self.num_batches:
            sys.stdout.write(msg)
            sys.stdout.flush()
        else:
            sys.stdout.write('\r')
            sys.stdout.flush()
            sys.stdout.write(msg[1:]+'\n')
            # logger.info(msg[1:])

    def display_summary(self):
        entries = [f"[{self.prefix}\b] "]
        entries += ['Time: ' + self.timer.summary()]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class NoOp:
    # https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op


def setup_logger(log_dir, rank, phase, to_console=True):
    if rank == 0:
        # File to save logger outputs
        log_file = Path(log_dir) / f'{phase}_rank{rank}.log'

        # Configure logger formats
        format  = '%(asctime)-15s %(message)s'
        datefmt = '%Y/%m/%d %H:%M:%S'
        logging.basicConfig(
            filename=str(log_file),
            datefmt=datefmt,
            format=format,
            level=logging.INFO
        )

        # Root logger to the file
        logger = logging.getLogger()

        # Stream handler to the console
        if to_console:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
            logger.addHandler(console)
    else:
        logger = NoOp()

    return logger


def create_logger_directories(cfg, rank, phase='train', write_cfg_to_file=True):
    #! NOTE: This function is called before processes are spawned
    # Where to save outputs (e.g., checkpoints)
    output_dir = Path(cfg.OUTPUT_DIR) / cfg.DATASET.DATANAME / cfg.EXP_NAME

    # Where to save logs
    time_str = time.strftime('%Y%m%d_%H_%M_%S')
    log_dir  = Path(cfg.LOG_DIR) / cfg.DATASET.DATANAME / cfg.EXP_NAME / f'{phase}_{time_str}'

    # Master rank make output directories
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=False)

    # Create logger
    logger = setup_logger(log_dir, rank, phase, to_console=True)

    if rank == 0 and write_cfg_to_file:
        with open(log_dir / 'config.txt', 'w') as f:
            f.write(str(cfg))

    logger.info(f'Outputs (e.g., checkpoints) are saved at:   {output_dir}')
    logger.info(f'Messages and tensorboard logs are saved at: {log_dir}')

    return logger, str(output_dir), str(log_dir)


# -----------------------------------------------------------------------
# Functions regarding 3D model loading
def load_camera_intrinsics(camera_json):
    with open(camera_json) as f:
        cam = json.load(f)

    # cam['cameraMatrix'] = np.array(cam['cameraMatrix'], dtype=np.float32)

    if 'distCoeffs' in cam.keys():
        cam['distCoeffs'] = np.array(cam['distCoeffs'], dtype=np.float32)

    # Compute horizontal FOV
    cam["horizontalFOV"] = 2.0 * np.arctan2(0.5 * cam["ppx"] * cam["Nu"], cam["fx"])

    return cam


# -----------------------------------------------------------------------
# Miscellaneous functions.
def set_seeds_cudnn(cfg, seed=None):
    if seed is None:
        seed = int(time.time())

    logger.info(f'Random seed: {seed}')

    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark     = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled       = cfg.CUDNN.ENABLED

        # empty any cached GPU memory
        torch.cuda.empty_cache()


def initialize_cuda(cfg, rank):
    if cfg.CUDA:
        torch.cuda.device(rank)
        device = torch.device('cuda', rank)
        if rank == 0:
            logger.info(f'GPU-accelerated training: ENABLED')
            logger.info(f'   • {torch.cuda.device_count()} x {torch.cuda.get_device_name(device)}')
    else:
        device = torch.device('cpu')
        if rank == 0:
            logger.info(f'GPU-accelerated training: DISABLED')

    return device


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum(p.numel() for p in model_parameters)


def get_max_cuda_memory(device=None) -> int:
    """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for
    a given device. By default, this returns the peak allocated memory since
    the beginning of this program.

    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.cuda.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
    torch.cuda.reset_peak_memory_stats()
    return int(mem_mb.item())