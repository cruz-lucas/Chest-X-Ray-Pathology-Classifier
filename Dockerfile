FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /project

COPY requirements.txt setup.py ./

# Installing dependencies (PIP)
RUN python3 -m ensurepip --upgrade && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    rm requirements.txt
