export CUDA_DIR=/usr/local/cuda
export ANACONDA_HOME=/home/car/3rd/anaconda2
export OPENCV_HOME=/home/car/3rd/opencv2
export CUDNN_HOME=/home/car/3rd/cudnn-v5
export PROTOBUF_HOME=/home/car/3rd/protobuf
export PATH=$PROTOBUF_HOME/bin:$ANACONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$OPENCV_HOME/lib:$ANACONDA_HOME/lib:$CUDA_DIR/lib64:/usr/lib64:$LD_LIBRARY_PATH

# glog
export GLOG_minloglevel=0
export GLOG_logtostderr=1
