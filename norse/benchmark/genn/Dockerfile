FROM nvidia/cuda

RUN apt update && apt install -y \
    git python3-pip swig 

# Checkout GeNN
RUN git clone https://github.com/genn-team/genn.git
WORKDIR /genn

# Setup env
RUN CUDA_PATH=/usr/local/cuda
RUN export CUDA_PATH
RUN PATH=$PATH:$CUDA_PATH/bin
RUN PATH=$PATH:`pwd`/bin
RUN export PATH

# Install PyGeNN
RUN make DYNAMIC=1 LIBRARY_DIRECTORY=`pwd`/pygenn/genn_wrapper/
RUN pip3 install numpy absl-py pandas matplotlib
RUN python3 setup.py develop

