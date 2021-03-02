# Define global args
ARG FUNCTION_DIR="/prediction-engine/"
ARG RUNTIME_VERSION="3.8"
ARG DISTRO_VERSION="3.12"

# Stage 1 - bundle base image + runtime
# Grab a fresh copy of the image and install GCC
FROM pytorch/pytorch:latest AS python-pytorch
# Install GCC (Alpine uses musl but we compile and link dependencies with GCC)
# RUN apt-get update && apt-get install -y \
#     libstdc++

# Stage 2 - build function and dependencies
FROM python-pytorch AS build-image
# Install aws-lambda-cpp build dependencies
RUN apt-get update -y  && apt-get install --no-install-recommends -y \
    python3.8 \
    python3-pip \
    python3.8-dev

RUN apt-get update && apt-get install -y \
        build-essential \
        libtool \
        autoconf \
        automake \
        make \
        cmake \
        python${RUNTIME_VERSION} \
    && apt-get clean all \
    && rm -r /var/lib/apt/lists/*

# RUN apt install python${RUNTIME_VERSION}-distutils && python${RUNTIME_VERSION} -m pip install --upgrade pip setuptools wheel
# Include global args in this stage of the build
ARG FUNCTION_DIR
ARG RUNTIME_VERSION
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}
# Copy handler function
COPY *.py ${FUNCTION_DIR}
COPY requirements.txt ${FUNCTION_DIR}
# Optional – Install the function's dependencies
# RUN /opt/conda/bin/conda install pytorch cpuonly -c pytorch --yes
# RUN /opt/conda/bin/conda install --file ${FUNCTION_DIR}requirements.txt --yes
RUN python${RUNTIME_VERSION} -m pip install -r ${FUNCTION_DIR}requirements.txt --target ${FUNCTION_DIR}
# RUN python${RUNTIME_VERSION} -m pip install torchtext==0.8 --target ${FUNCTION_DIR}
# Install Lambda Runtime Interface Client for Python
RUN python${RUNTIME_VERSION} -m pip install awslambdaric --target ${FUNCTION_DIR}

# Stage 3 - final runtime image
# Grab a fresh copy of the Python image
FROM python-pytorch
# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}
# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}
# (Optional) Add Lambda Runtime Interface Emulator and use a script in the ENTRYPOINT for simpler local runs
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
COPY entry.sh /
RUN chmod 755 /usr/bin/aws-lambda-rie /entry.sh
ENTRYPOINT [ "/entry.sh" ]
CMD [ "predictions.main" ]