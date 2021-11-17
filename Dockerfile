# syntax=docker/dockerfile:1
FROM debian:latest

WORKDIR /project

# Installing python and necessary software
RUN apt update && apt upgrade && \
    apt install -y python3 g++ make python3-pip curl git nano

# Copying necessary files
# COPY ./ ./

# Installing dependencies (PIP)
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

EXPOSE 8888
