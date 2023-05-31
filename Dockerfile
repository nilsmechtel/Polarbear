# Use an official tensorflow runtime as a parent image
# - Python 3.6.9
# - Tensorflow 1.15.4
FROM tensorflow/tensorflow:1.15.4-gpu-py3

# Update system packages
# GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
RUN apt-key adv \
        --keyserver keyserver.ubuntu.com \
        --recv-keys A4B469963BF863CC && \
    apt-get update

# Install requirements
# Note: swagger-spec-validator>2.7.6 requires Python3.7 but Python3.6 is installed
RUN pip install --upgrade pip && \
    pip install scikit-learn==0.23.2 && \
    pip install pandas==1.1.5 && \
    pip install neptune-client==0.15.2 && \
    pip install swagger-spec-validator==2.7.6
    
# Set the working directory to /home
WORKDIR /home
