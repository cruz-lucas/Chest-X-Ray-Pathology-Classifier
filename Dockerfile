FROM conda/miniconda3

WORKDIR /project

RUN apt-get update && apt-get install -y make ffmpeg libsm6 libxext6

COPY requirements.txt setup.py ./

# Installing dependencies (PIP)
RUN python3 -m ensurepip --upgrade && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    rm requirements.txt
