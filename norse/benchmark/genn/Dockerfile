FROM nvidia/cuda:11.0-devel

RUN apt update && apt install -y \
    git python3-pip swig 

# Checkout GeNN
RUN git clone https://github.com/genn-team/genn.git
WORKDIR /genn

# Setup env
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$PATH:$CUDA_PATH/bin
ENV PATH=$PATH:/genn/bin

# Install PyGeNN
RUN "make DYNAMIC=1 LIBRARY_DIRECTORY=`pwd`/pygenn/genn_wrapper/"
RUN "pip3 install numpy absl-py pandas matplotlib"
RUN "python3 setup.py develop"

