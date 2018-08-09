#!/usr/bin/env bash

cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

export PYTHONPATH=$root_dir/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/.apps/cudnn-v5/lib64:$HOME/.apps/opencv2/lib:$HOME/.apps/anaconda2/lib:/usr/local/cuda-9.1/lib64:/usr/lib64:$LD_LIBRARY_PATH

redo=false
dataset_name=lot
data_root_dir=$HOME/data/${dataset_name}
mapfile=${root_dir}/data/${dataset_name}/labelmap.prototxt
anno_type=detection
label_type=json
db=lmdb
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
    extra_cmd="$extra_cmd --redo"
fi
for subset in train2500 val2500
do
    python ${root_dir}/scripts/create_annoset.py \
           --anno-type=$anno_type \
           --label-type=$label_type \
           --label-map-file=$mapfile \
           --min-dim=$min_dim \
           --max-dim=$max_dim \
           --resize-width=$width \
           --resize-height=$height \
           --check-label \
           $extra_cmd \
           $data_root_dir \
           $root_dir/data/$dataset_name/$subset.txt \
           $data_root_dir/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name 2>&1 | tee $root_dir/data/$dataset_name/$subset.log
done
