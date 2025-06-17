FROM pytorchlightning/pytorch_lightning:base-cuda-py3.12-torch2.6-cuda12.4.1
WORKDIR /temp
COPY requirements.txt .
RUN pip install --upgrade-strategy only-if-needed -r requirements.txt
RUN apt-get update && \
    apt-get install -y git

WORKDIR /workspace
RUN rm -rf /temp/*
