FROM ubuntu:16.04

RUN apt update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
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
        # gpg-agent \ # necessary for cloud-sdk if > Ubuntu 16.04
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install gcloud / gsutil
ENV PATH /google-cloud-sdk/bin:$PATH
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y
VOLUME ["/root/.config"]

WORKDIR /app

# Here for caching.
ADD staging/requirements.txt /app/requirements.txt
ADD staging/minigui/requirements.txt /app/minigui/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install -r /app/minigui/requirements.txt
RUN pip3 install "tensorflow>=1.7,<1.8"

ADD staging/ /app

WORKDIR /app

CMD ["/bin/bash", "minigui/fetch-and-run.sh"]
