FROM python:3.9-slim

RUN apt update && apt install -y python3-pip build-essential
RUN rm -rf /var/lib/apt/lists/

WORKDIR /norse

COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -e .