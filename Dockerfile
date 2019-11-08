FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

WORKDIR /myelin

COPY . .

RUN pip install -r requirements.txt --no-cache-dir