FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt update && apt-get install -y --no-install-recommends \
        build-essential \
        bc \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        python3 \
        python3-numpy \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-setuptools \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

ADD staging/requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install "tensorflow-gpu==1.15.0"

ADD staging/ /app

CMD ["/bin/sh", "evaluator_py_wrapper.sh"]
