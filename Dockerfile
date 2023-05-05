# Use an official tensorflow runtime as a parent image
# - Python 3.6.9
# - Tensorflow 1.15.4
FROM tensorflow/tensorflow:1.15.4-gpu-py3

# Install requirements
# Note: swagger-spec-validator>2.7.6 requires Python3.7 but Python3.6 is installed
RUN pip install --upgrade pip && \
    pip install scikit-learn==0.23.2 && \
    pip install pandas==1.1.5 && \
    pip install neptune-client==0.15.2 && \
    pip install swagger-spec-validator==2.7.6
    
# Set the working directory to /home
WORKDIR /home
