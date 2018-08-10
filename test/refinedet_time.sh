#!/usr/bin/env bash

cd /home/fzhou/code/RefineDet
./build/tools/caffe \
    time \
    --gpu 0 \
    --model="models/VGGNet/v1/refinedet_vgg16_320x320/deploy.prototxt" \
    # --model="models/VGGNet/v1/refinedet_vgg16_512x512_ft/deploy.prototxt" \
