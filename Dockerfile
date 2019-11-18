FROM nvidia/cuda:latest

RUN apt update && apt install -y python3-setuptools
RUN rm -rf /var/lib/apt/lists/

WORKDIR /myelin

COPY . .

RUN python3 setup.py install
