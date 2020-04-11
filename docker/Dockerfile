FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV PATH /usr/lib/ccache:$PATH

RUN set -ex \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    software-properties-common \
    language-pack-ja-base language-pack-ja \
    xvfb ffmpeg python3-opengl zlib1g-dev cmake \
    ccache curl g++ gfortran git libhdf5-dev wget xz-utils libopencv-dev python-dev unzip \
  && add-apt-repository -y ppa:jonathonf/python-3.6 \
  && apt-get update \
  && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3-tk \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

RUN ln -sf /usr/bin/python3.6 /usr/bin/python3 \
  && ln -sf /usr/bin/python3.6 /usr/bin/python

RUN python -m pip install -U pip \
  && pip install -U --no-cache-dir \
    wheel setuptools \
    numpy \
    scipy \
    matplotlib\
    gym \
    gym[atari]

RUN apt-get update \
    && apt-get --yes --no-install-recommends install \
        python3 python3-dev \
        python3-pip python3-venv python3-wheel python3-setuptools \
        build-essential \
        graphviz git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install gym[atari]

WORKDIR /work
ENTRYPOINT xvfb-run -s "-screen 0 1400x900x24" bash

# RUN apt-get -y -q install python-opencv
# RUN pip install -U "setuptools"
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip
# RUN pip install -U "torch==0.4.0"
# RUN pip install -U "torchvision"
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl 
RUN pip install torchvision


RUN pip install -U "cython==0.25"
RUN pip install -U "h5py"
RUN pip install -U "scipy"
RUN pip install -U "matplotlib"
RUN pip install -U "hacking"
RUN pip install -U "nose"
RUN pip install -U "mock"
RUN pip install -U "coverage"
RUN pip install -U "pillow"
RUN pip install -U "setproctitle"

# RUN pip install torch torchvision
# RUN pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
# RUN pip install torch-0.3.1-cp36-cp36m-linux_x86_64.whl
# RUN pip install -U "torch==0.3.1"
# RUN pip install -U "torchvision==0.2.0"

WORKDIR /
ENV OPENCV_VERSION="3.4.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DENABLE_AVX=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.6 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.6) \
  -DPYTHON_INCLUDE_DIR=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.6 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}


