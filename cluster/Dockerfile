FROM ubuntu:16.04

RUN apt update && apt-get install -y --no-install-recommends \
        build-essential \
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
        openjdk-8-jdk \
        openjdk-8-jre-headless \
				fuse \
				python3 \
				python3-numpy \
				python3-dev \
				python3-pip \
				python3-wheel \
				python3-setuptools \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.5.4
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.

RUN git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout r1.4
WORKDIR /tensorflow

ENV CI_BUILD_PYTHON python3

# --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
RUN ln -s /usr/bin/python3 /usr/bin/python  && tensorflow/tools/ci_build/builds/configured CPU \
    bazel build -c opt --copt=-march="haswell" \ 
        tensorflow/tools/pip_package:build_pip_package --verbose_failures && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip3 --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
    rm -rf /tmp/pip && \
    rm -rf /root/.cache

WORKDIR /

# RUN apt-get update && apt-get install -f -y fuse uuid-runtime pkg-config python3 python3-pip libffi6 libfreetype6 libjpeg62-turbo libopenblas-base libffi-dev libfreetype6-dev libpng-dev libjpeg62-turbo-dev libopenblas-dev curl

RUN curl -v -L -O https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v0.23.0/gcsfuse_0.23.0_amd64.deb && \
    dpkg -i gcsfuse_0.23.0_amd64.deb || echo fixing dependencies && \
        apt-get update && apt-get install -f -y uuid-runtime


ADD staging/ /app

RUN pip3 install -r /app/requirements.txt

WORKDIR /app

ENTRYPOINT ["./player_wrapper.sh"]
