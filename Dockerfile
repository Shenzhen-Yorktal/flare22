# Here is an example of a Dockerfile to use. Please make sure this file is placed to the same folder as run_inference.py file and directory model/ that contains your training weights.

# FROM ubuntu:latest
FROM nvcr.io/nvidia/pytorch:21.02-py3

# Install some basic utilities and python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install SimpleITK==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
# install nnunet
RUN pip install nnunet -i https://pypi.tuna.tsinghua.edu.cn/simple


# RUN pip3 install numpy simpleitk -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the folder with your pretrained model here to /model folder within the container. This part is skipped here due to simplicity reasons
# ADD model /model/
ADD parameters /parameters/
ADD nnUNet /nnUNet/
ADD run_inference.py ./
ADD predict_1by1.py ./
ADD predict.sh ./


# RUN groupadd -r myuser -g 433 && \
#     useradd -u 431 -r -g myuser -s /sbin/nologin -c "Docker image user" myuser

RUN mkdir -p /workspace/inputs && mkdir -p /workspace/outputs 
RUN pip install -e /nnUNet/

# 
# USER myuser
# CMD python3 ./run_inference.py

