ARG project
#FROM gcr.io/$project/cc-base:v17-testing
from base-build-manual2

RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

RUN apt-get install python3 python3-pip -y
RUN pip3 install absl-py

COPY staging/ /app
WORKDIR /app
RUN bazel build -c opt --define=tf=1 --define=bt=1 cc/eval

COPY evaluator_cc_wrapper.sh /app

CMD ["/bin/bash", "evaluator_cc_wrapper.sh"]
