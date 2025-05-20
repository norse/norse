FROM python:3.12-slim

RUN apt update && apt install -y git python3-pip build-essential
RUN rm -rf /var/lib/apt/lists/

WORKDIR /norse

COPY . .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .