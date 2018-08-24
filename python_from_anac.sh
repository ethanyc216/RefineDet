# !/bin/bash
#
# Install anaconda2.
#
# History
#   create  -  Feng Zhou (zhfe99@gmail.com), 2015-06

set -e
set -x

# set env
export APPS=${HOME}/.apps
[ ! -d ${APPS} ] && mkdir -p $APPS
export PYTHON_HOME=${APPS}/anaconda2
export PATH=${PYTHON_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:$LD_LIBRARY_PATH

# install anaconda2
cd /tmp
# wget https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh
cp /mnt/soulfs2/fzhou/software/Anaconda2-5.2.0-Linux-x86_64.sh .
[ -d $PYTHON_HOME ] && rm -fr $PYTHON_HOME
bash Anaconda2-5.2.0-Linux-x86_64.sh -p $PYTHON_HOME -b

# install common package
pip install easydict
pip install protobuf
pip install argparse
