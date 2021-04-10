FROM ubuntu:18.04

ARG DOCKER_VER
ENV DOCKER_BUILT_VER ${DOCKER_VER}}

ARG NUM_BUILD_CORES=8
ENV NUM_BUILD_CORES ${NUM_BUILD_CORES}

COPY /test_usage /

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y install --no-install-recommends \ 
        software-properties-common \
        automake \
        autoconf \
        apt-utils \
        bc \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        flac \
        ffmpeg \
        gawk \
        gfortran \
        git \
        libtool \
        libsndfile1-dev \
        python2.7 \
        python3-pip \
        python3.7 \
        python3-setuptools \
        python3-wheel \
        sox \
        subversion \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        llvm-9 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN export LLVM_CONFIG=/usr/bin/llvm-config-9 
    
RUN apt-get update 

RUN apt install -y python3.7-dev 

RUN python3.7 -m pip install -r requirements.txt

RUN git clone https://github.com/espnet/espnet
RUN git clone https://github.com/kaldi-asr/kaldi

RUN cd espnet/tools \
    ln -s ../kaldi-asr

RUN cd espnet \
python3.7 -m pip install -e .

WORKDIR /