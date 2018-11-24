#!/bin/bash

CAFFE_HOME=/jobs-docker/RefineDet

old_model=/jobs-docker/model/ResNet-32-model.caffemodel

# glog
export GLOG_minloglevel=0
export GLOG_logtostderr=1

${CAFFE_HOME}/build/tools/caffe \
    train \
    --gpu 1,2 \
    --solver=solver.prototxt \
    --weights=${old_model} \
    2>&1 | tee train.log
