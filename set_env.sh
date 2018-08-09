export PYTHONPATH=$PWD/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/.apps/cudnn-v5/lib64:$HOME/.apps/opencv2/lib:$HOME/.apps/anaconda2/lib:/usr/local/cuda-9.1/lib64:/usr/lib64:$LD_LIBRARY_PATH

# glog
export GLOG_minloglevel=0
export GLOG_logtostderr=1
