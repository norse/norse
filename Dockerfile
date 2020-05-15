FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN apt update && apt install -y python3-setuptools
RUN rm -rf /var/lib/apt/lists/

WORKDIR /norse

COPY . .

RUN python3 setup.py install
