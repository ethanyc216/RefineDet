#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
import os
os.chdir('..')
caffe_root = './'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

coco_net = caffe.Net(caffe_root + 'models/VGGNet/coco/refinedet_vgg16_320x320/deploy.prototxt',
                     caffe_root + 'models/VGGNet/coco/refinedet_vgg16_320x320/coco_refinedet_vgg16_320x320_final.caffemodel',
                     caffe.TEST)

# voc_net = caffe.Net(caffe_root + 'models/VGGNet/VOC0712/refinedet_vgg16_320x320/deploy.prototxt',
#                     caffe_root + 'models/VGGNet/VOC0712/refinedet_vgg16_320x320/VOC0712_refinedet_vgg16_320x320_final.caffemodel')

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load MS COCO model specs
file = open(caffe_root + 'models/VGGNet/coco/refinedet_vgg16_320x320/deploy.prototxt', 'r')
coco_netspec = caffe_pb2.NetParameter()
text_format.Merge(str(file.read()), coco_netspec)

# load MS COCO labels
coco_labelmap_file = caffe_root + 'data/coco/labelmap_coco.prototxt'
file = open(coco_labelmap_file, 'r')
coco_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), coco_labelmap)

# load CCAR model specs
file = open(caffe_root + 'models/VGGNet/ccar/refinedet_vgg16_320x320/deploy.prototxt', 'r')
ccar_netspec = caffe_pb2.NetParameter()
text_format.Merge(str(file.read()), ccar_netspec)

# load CCAR labels
ccar_labelmap_file = caffe_root + 'data/ccar/labelmap_ccar.prototxt'
file = open(ccar_labelmap_file, 'r')
ccar_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), ccar_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

map_file = caffe_root + 'data/ccar/coco_ccar_map.txt'
if not os.path.exists(map_file):
    print('{} does not exist'.format(map_file))

maps = np.loadtxt(map_file, str, delimiter=',')
for m in maps:
    [coco_label, ccar_label, name] = m
    coco_name = get_labelname(coco_labelmap, int(coco_label))[0]
    ccar_name = get_labelname(ccar_labelmap, int(ccar_label))[0]
    assert ccar_name == name
    print('{}, {}'.format(coco_name, ccar_name))

def sample_param(src_param, src_num_classes, dst_num_classes, num_bboxes, maps):
    src_shape = src_param.shape
    assert src_shape[0] == src_num_classes * num_bboxes
    if len(src_shape) == 4:
        dst_shape = (dst_num_classes * num_bboxes, src_shape[1], src_shape[2], src_shape[3])
    else:
        dst_shape = dst_num_classes * num_bboxes
    try:
        dst_param = np.zeros(dst_shape)
    except:
        import pdb; pdb.set_trace()
    for i in xrange(0, num_bboxes):
        for m in maps:
            [src_label, dst_label, name] = m
            src_idx = i * src_num_classes + int(src_label)
            dst_idx = i * dst_num_classes + int(dst_label)
            dst_param[dst_idx,] = src_param[src_idx,]
    return dst_param

mbox_source_layers = ['P3', 'P4', 'P5', 'P6']
num_bboxes = [3, 3, 3, 3]
assert len(mbox_source_layers) == len(num_bboxes)
num_ccar_classes = 2
num_coco_classes = 81

for i in xrange(0, len(mbox_source_layers)):
    mbox_source_layer = mbox_source_layers[i]
    mbox_priorbox_layer = '{}_mbox_priorbox'.format(mbox_source_layer)
    mbox_loc_layer = '{}_mbox_loc'.format(mbox_source_layer)
    mbox_conf_layer = '{}_mbox_conf'.format(mbox_source_layer)
    num_bbox = num_bboxes[i]
    for j in xrange(0, len(coco_netspec.layer)):
        layer = coco_netspec.layer[j]
        if mbox_priorbox_layer == layer.name:
            ccar_netspec.layer[j].prior_box_param.CopyFrom(layer.prior_box_param)
        if mbox_loc_layer == layer.name:
            ccar_netspec.layer[j].convolution_param.num_output = num_bbox * 4
        if mbox_conf_layer == layer.name:
            ccar_netspec.layer[j].convolution_param.num_output = num_bbox * num_ccar_classes

new_ccar_model_dir = caffe_root + 'models/VGGNet/ccar/refinedet_vgg16_320x320_coco'
if not os.path.exists(new_ccar_model_dir):
    os.makedirs(new_ccar_model_dir)
# del ccar_netspec.layer[-1]
new_ccar_model_def_file = '{}/deploy.prototxt'.format(new_ccar_model_dir)
with open(new_ccar_model_def_file, 'w') as f:
    print(ccar_netspec, file=f)

ccar_net_new = caffe.Net(new_ccar_model_def_file, caffe.TEST)
new_ccar_model_file = '{}/coco_refinedet_vgg16_320x320.caffemodel'.format(new_ccar_model_dir)

for layer_name, param in coco_net.params.iteritems():
    if 'mbox' in layer_name and 'P' in layer_name:
        continue
    else:
        for i in xrange(0, len(param)):
            ccar_net_new.params[layer_name][i].data.flat = coco_net.params[layer_name][i].data.flat

for i in xrange(0, len(mbox_source_layers)):
    layer = mbox_source_layers[i]
    num_bbox = num_bboxes[i]
    conf_layer = '{}_mbox_conf'.format(layer)
    ccar_net_new.params[conf_layer][0].data.flat = sample_param(coco_net.params[conf_layer][0].data,
                                                                len(coco_labelmap.item),
                                                                len(ccar_labelmap.item),
                                                                num_bbox,
                                                                maps)
    ccar_net_new.params[conf_layer][1].data.flat = sample_param(coco_net.params[conf_layer][1].data,
                                                      len(coco_labelmap.item), len(ccar_labelmap.item), num_bbox, maps)
    loc_layer = '{}_mbox_loc'.format(layer)
    ccar_net_new.params[loc_layer][0].data.flat = coco_net.params[loc_layer][0].data.flat
    ccar_net_new.params[loc_layer][1].data.flat = coco_net.params[loc_layer][1].data.flat
ccar_net_new.save(new_ccar_model_file)
