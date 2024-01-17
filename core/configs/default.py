'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''


from os.path import join
from yacs.config import CfgNode as CN

_C = CN()

# ------------------------------------------------------------------------------ #
# Basic settings
# ------------------------------------------------------------------------------ #
_C.ROOT       = '/path/to/project/root/directory'                   # Project root directory
_C.OUTPUT_DIR = 'output'                                            # Name of the folder to save training outputs
_C.LOG_DIR    = 'log'                                               # Name of the folder to save trainings logs
_C.EXP_NAME   = 'expNameTemp'                                       # Current experiment name

# Basic settings
_C.LOG_TENSORBOARD = False
_C.CUDA            = False                                          # Use GPU?
_C.AMP             = False                                          # Use mixed precision?
_C.AMP_DTYPE       = "float16"
_C.AUTO_RESUME     = True                                           # Pick up from the last available training session?
_C.PIN_MEMORY      = True
_C.SEED            = None                                           # Random seed. If None, seed is determined based on computer time
_C.VERBOSE         = False

# cudNN related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK     = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED       = True

# Distributed training
_C.DIST = CN()
_C.DIST.RANK = 0
_C.DIST.BACKEND = 'nccl'
_C.DIST.MULTIPROCESSING_DISTRIBUTED = False

# ------------------------------------------------------------------------------ #
# Dataset-related parameters
# ------------------------------------------------------------------------------ #
_C.DATASET = CN()

# - Basic directory & files
#   ROOT/DATANAME
#   - CAMERA
#   - IMAGE_DIR
_C.DATASET.ROOT         = '/home/jeffpark/SLAB/Dataset'
_C.DATASET.DATANAME     = 'satnet'
_C.DATASET.CAMERA       = 'camera.json'
_C.DATASET.IMAGE_DIR    = 'images'
_C.DATASET.MASK_DIR     = 'masks'

# - I/O
_C.DATASET.IMAGE_SIZE         = [400, 300]
_C.DATASET.NUM_POINTS_ON_MESH = 2000
_C.DATASET.NUM_POINTS_IN_MESH = 10000

# - Files
_C.DATASET.SPLIT_CSV = 'splits.csv'

# ------------------------------------------------------------------------------ #
# Model-related parameters
# ------------------------------------------------------------------------------ #
_C.MODEL = CN()
_C.MODEL.PRETRAIN_FILE = None

# Model spec.
_C.MODEL.LATENT_DIM         = 128
_C.MODEL.HIDDEN_DIM         = 256
_C.MODEL.NUM_MAX_PRIMITIVES = 5
_C.MODEL.APPLY_TAPER        = True

# Render
_C.MODEL.ICOSPHERE_LEVEL = 3
_C.MODEL.RENDER_SIGMA    = 1e-4

# Ablation
_C.MODEL.USE_TRUE_POSE_RENDER = False
_C.MODEL.USE_DUAL_SQ          = True

# ------------------------------------------------------------------------------ #
# Model-related parameters
# ------------------------------------------------------------------------------ #
_C.LOSS = CN()
_C.LOSS.RECON_TYPE   = []
_C.LOSS.RECON_WEIGHT = []
_C.LOSS.POSE_TYPE    = []
_C.LOSS.POSE_WEIGHT  = []
_C.LOSS.REG_TYPE     = []
_C.LOSS.REG_WEIGHT   = []
_C.LOSS.SHARPNESS    = 10
_C.LOSS.BETA_OVERLAP = 2.0

# ------------------------------------------------------------------------------ #
# Training-related parameters
# ------------------------------------------------------------------------------ #
_C.TRAIN = CN()

# - Learning rate & scheduler
_C.TRAIN.LR        = 0.001
_C.TRAIN.SCHEDULER = 'step'
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP   = [90, 110]

# - Optimizer
_C.TRAIN.OPTIMIZER = 'SGD'                                  # Optimizer name. Must be same as PyTorch optimizer name
_C.TRAIN.WD        = 0.0001                                 # Weight decay factor
_C.TRAIN.EPS       = 1e-5
_C.TRAIN.GAMMA1    = 0.9                                    # Momentum factor
_C.TRAIN.GAMMA2    = 0.999                                  # Secondary momentum factor for Adam optimizers

# - Epochs
_C.TRAIN.BEGIN_EPOCH     = 0
_C.TRAIN.END_EPOCH       = 100
# _C.TRAIN.STEPS_PER_EPOCH = 500
_C.TRAIN.VALID_FREQ      = 20
_C.TRAIN.VALID_FRACTION  = None                              # Fraction of validation set on which inference is conducted

# - Batches
_C.TRAIN.BATCH_SIZE_PER_GPU = 16                            # Batch size PER GPU, NOT across all GPUs!
_C.TRAIN.SHUFFLE            = True
_C.TRAIN.WORKERS            = 4

# ------------------------------------------------------------------------------ #
# Test-related parameters
# ------------------------------------------------------------------------------ #
_C.TEST = CN()

# - Batches
_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.NUM_REPEATS        = 1


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # Add any processing here
    cfg.DATASET.CAMERA = join(cfg.DATASET.ROOT, cfg.DATASET.DATANAME, cfg.DATASET.CAMERA)

    assert 'spe3r' in cfg.DATASET.DATANAME, 'only spe3r variants are supported for datasets'

    cfg.freeze()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)