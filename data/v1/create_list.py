import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys

HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

# If true, re-create all list files.
redo = True
# The root directory which holds all information of the dataset.
data_dir = "{}/data/v1".format(HOMEDIR)
# The directory name which holds the image sets.
imgset_dir = "ImageSets"
# The direcotry which contains the images.
img_dir = "images"
img_ext = "jpg"
# The directory which contains the annotations.
anno_dir = "Annotations"
anno_ext = "json"

train_list_file = "{}/train.txt".format(CURDIR)
minival_list_file = "{}/val.txt".format(CURDIR)

# Create training set.
if redo or not os.path.exists(train_list_file):
    datasets = ["train"]
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n")
                subset = dataset
                img_file = "{}/{}.{}".format(subset, name, img_ext)
                assert os.path.exists("{}/{}".format(data_dir, img_file)), \
                        "{}/{} does not exist".format(data_dir, img_file)
                anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
                assert os.path.exists("{}/{}".format(data_dir, anno_file)), \
                        "{}/{} does not exist".format(data_dir, anno_file)
                img_files.append(img_file)
                anno_files.append(anno_file)
    # Shuffle the images.
    idx = [i for i in xrange(len(img_files))]
    shuffle(idx)
    with open(train_list_file, "w") as f:
        for i in idx:
            f.write("{} {}\n".format(img_files[i], anno_files[i]))

if redo or not os.path.exists(minival_list_file):
    datasets = ["val"]
    subset = "val"
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n")
                img_file = "{}/{}.{}".format(subset, name, img_ext)
                assert os.path.exists("{}/{}".format(data_dir, img_file)), \
                        "{}/{} does not exist".format(data_dir, img_file)
                anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
                assert os.path.exists("{}/{}".format(data_dir, anno_file)), \
                        "{}/{} does not exist".format(data_dir, anno_file)
                img_files.append(img_file)
                anno_files.append(anno_file)
    with open(minival_list_file, "w") as f:
        for i in xrange(len(img_files)):
            f.write("{} {}\n".format(img_files[i], anno_files[i]))
