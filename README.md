# Rapid Abstraction of Spacecraft 3D Structure from Single 2D Image

This repository is developed by Tae Ha "Jeff" Park at [Space Rendezvous Laboratory (SLAB)](https://slab.stanford.edu) of Stanford University.

## Updates

- [2024/03/18] The udpated dataset (v1.1) is now available at the [Stanford Digital Repository](https://purl.stanford.edu/pk719hm4806). This version corrects bugs with the pose labels in the initial deposit, so the users are encouraged to download the updated version. Please see `UPDATES.md` inside the new dataset for a detailed update log.
- [2024/09/22] Updated the link to the pre-trained models.

## Introduction

This is the official PyTorch implementation of the paper titled [Rapid Abstraction of Spacecraft 3D Structure from Single 2D Image](https://arc.aiaa.org/doi/10.2514/6.2024-2768).

### Abstract

This paper presents a Convolutional Neural Network (CNN) to simultaneously abstract the 3D structure of the target space resident object and estimate its pose from a single 2D image. Specifically, the CNN predicts from a single image of the target a unit-size assembly of superquadric primitives which can individually describe a wide range of simple 3D shapes (e.g., cuboid, ellipsoid) using only a few parameters. The proposed training pipeline employs various types of supervision in both 2D and 3D spaces to fit an assembly of superquadrics to man-made satellite structures. In order to avoid numerical instability encountered when evaluating superquadrics, this work proposes a novel, numerically stable algorithm based on dual superquadrics to evaluate a point on the surface of and inside a superquadric for all shape parameters. Furthermore, in order to train the CNN, this work also introduces SPE3R, a novel dataset comprising 64 different satellite models and 1,000 images, binary masks and pose labels for each model. The experimental studies reveal that the proposed CNN can be trained to reconstruct accurate superquadric assemblies when tested on unseen images of known models and capture high-level structures of the unknown models most of the time despite having been trained on an extremely small dataset.

## Install

This repository uses [poetry](https://python-poetry.org) to manage virtual environment and library dependencies. It is developed and tested on Ubuntu 22.04 and trained on a single NVIDIA GeForce RTX 4090 24GB GPU.

1. Install [poetry](https://python-poetry.org/docs/#installation) and [create a virtual environment](https://python-poetry.org/docs/managing-environments/) for `python 3.10` as shown in the `pyproject.toml` file.

2. Install dependencies specified in `pyproject.toml` by running

    ``` bash
    poetry install
    ```

3. Separately install `kaleido` via

    ``` bash
    pip install -U kaleido
    ```

    which is necessary to use [plotly](https://plotly.com/python/) for visualization. It seems there is no poetry support for it yet.

4. Compile the extension module via

    ``` bash
    python setup.py build_ext --inplace
    ```

5. [Optional] Get pre-trained model from [here](https://1drv.ms/f/c/fa28139a835eeb46/Evpp5SltMNNFqX_W26jaCzAB_UF6knvqKmkF-143sSAMVw). It is trained with `M = 8` primitives for both RGH and grayscale image inputs.

## SPE3R

First, obtain the SPE3R dataset from [Stanford Digital Repository](https://purl.stanford.edu/pk719hm4806) and place it at `ROOT`. You can use `tools/get_spe3r.sh` to unzip all files

``` bash
sh tools/get_spe3r.sh /path/to/dataset
```

Note that this dataset is slightly different from the one used in the experiments reported in the paper. There is a small difference in model composition for the `validation` set.

## Scripts

### Preprocessing

`tools/preprocess.py` will create 100,000 occupancy labels (`occupancy_points.npz`) and surface points (`surface_points.npz`) for each model. They will also save images of these points for visual verification.

``` bash
python tools/preprocess.py --cfg experiments/config.yaml
```

### Training

You will want to set the following before running the script.

| Variable                   | Details                       |
|----------------------------|-------------------------------|
| `ROOT`                     | Location of repo              |
| `DATASET.ROOT`             | Location of dataset           |
| `EXP_NAME`                 | Name of this training session |
| `MODEL.NUM_MAX_PRIMITIVES` | Number of primitives (FIXED)  |

You can also change `LOSS.RECON_TYPE`, `LOSS.POSE_TYPE`, `LOSS.REG_TYPE` to change the main supervised losses, pose losses, and regularizations to be used in the training.

The training can then be done via

``` bash
python tools/train.py --cfg experiments/config.yaml
```

### Evaluation

Below command will save the output superquadric mesh and input image to `SAVE_DIR`. Unless `--modelidx` and `--imageidx` are explicited provided, it will evaluate on a random image of a random model for the given data split.

``` bash
SPLIT=validation
SAVE_DIR=figures
NUM_PRIM=8
PRETRAIN=output/model_m8.pth.tar

python tools/evaluate.py --cfg experiments/config.yaml --split ${SPLIT} --save_dir ${SAVE_DIR} \
        MODEL.NUM_MAX_PRIMITIVES ${NUM_PRIM} MODEL.PRETRAIN_FILE ${PRETRAIN}
```

## License

This repository is released under the MIT License (see `LICENSE.md` for more details).

## Citation

```
@inbook{park_2024_scitech_spe3r,
    author = {Park, Tae Ha and D'Amico, Simone},
    title = {Rapid Abstraction of Spacecraft 3D Structure from Single 2D Image},
    booktitle = {AIAA SCITECH 2024 Forum},
    chapter = {},
    pages = {},
    doi = {10.2514/6.2024-2768},
}
```
