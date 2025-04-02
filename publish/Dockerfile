FROM python:3.11-slim

RUN apt update && apt install -y python3-pip build-essential git
RUN rm -rf /var/lib/apt/lists/

WORKDIR /norse

COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -e .
