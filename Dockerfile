FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
LABEL org.opencontainers.image.source https://github.com/flyteorg/flytesnacks

WORKDIR /root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

# Set your wandb API key and user name. Get the API key from https://wandb.ai/authorize.
# ENV WANDB_API_KEY <api_key>
# ENV WANDB_USERNAME <user_name>

# Install the AWS cli for AWS support
RUN pip install awscli pdm

# Install gcloud for GCP
RUN apt-get update && apt-get install -y make build-essential libssl-dev curl

# Virtual environment
ENV VENV /opt/venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Copy the actual code
COPY . /root/

# Install Python dependencies
COPY pdm.lock /root
RUN pdm install -p /root/

# This tag is supplied by the build script and will be used to determine the version
# when registering tasks, workflows, and launch plans
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag