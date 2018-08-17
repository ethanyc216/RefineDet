# !/bin/bash
#
# Install opencv.
#
# History
#   create  -  Feng Zhou (zhfe99@gmail.com), 2018-03

set -e

export APPS=${HOME}/.apps
[ ! -d ${APPS} ] && mkdir -p $APPS

# download source
cd /tmp
# wget https://github.com/opencv/opencv/archive/2.4.13.6.zip
cp /mnt/soulfs2/fzhou/software/opencv-2.4.13.6.zip .
[ -d opencv-2.4.13.6 ] && rm -fr opencv-2.4.13.6
unzip -q opencv-2.4.13.6.zip

# make build
cd opencv-2.4.13.6
[ -d build ] && rm -fr build
mkdir build
cd build

# cmake
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${APPS}/opencv2 \
    -DBUILD_PNG=ON \
    -DWITH_CUDA=OFF \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
    -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
    .. >cmake.log 2>cmake.err

# make
make -j8
make install
