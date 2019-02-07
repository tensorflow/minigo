FROM nvidia/opencl:runtime-ubuntu16.04
RUN apt-get update --no-upgrade -yq && \
    apt-get install --no-upgrade -yq curl git clinfo cmake g++ libboost-dev libboost-program-options-dev libboost-filesystem-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev qtbase5-dev python-virtualenv lsb-release

RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y


# Install Leela.
RUN mkdir /app && cd /app && \
    git clone https://github.com/leela-zero/leela-zero.git --branch next . && \
    git submodule update --init --recursive && \
    sed -i 's!//#define USE_TUNER!#define USE_TUNER!' src/config.h && \
    mkdir build && cd build && \
    cmake .. && cmake --build . -- -j 8
    # ./tests

RUN cp /app/build/leelaz /leelaz


WORKDIR /
RUN virtualenv -p /usr/bin/python2 mg_venv
RUN . mg_venv/bin/activate \
  && pip install gomill
# ringmaster now available at /mg_venv/bin/ringmaster


# It'd be nice if this was a flagfile.
ADD staging/p100-lz-tuning /root/.local/share/leela-zero/leelaz_opencl_tuning
ADD staging/ringmaster_wrapper.sh /


CMD ["/bin/sh", "ringmaster_wrapper.sh"]
