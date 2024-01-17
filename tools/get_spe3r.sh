#!/bin/bash
#
# Copyright (c) 2024 SLAB Group
# Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)

# Edit this if needed
DATA_DIR=$1

# -=-=-=-=-=- First, make sure the dataset is downloaded and unzip it -=-=-=-=-=- #
unzip "${DATA_DIR}.zip" -d ${DATA_DIR}

# -=-=-=-=-=- Unzip images & masks, remove .zip files -=-=-=-=-=- #
for MODEL in ${DATA_DIR}/*;
do
    echo $MODEL

    mkdir ${MODEL}/images
    mkdir ${MODEL}/masks

    unzip ${MODEL}/images.zip -d ${MODEL}/images
    unzip ${MODEL}/masks.zip  -d ${MODEL}/masks

    rm ${MODEL}/images.zip
    rm ${MODEL}/masks.zip
done