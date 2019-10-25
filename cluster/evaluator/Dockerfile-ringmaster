ARG project
from base-build-manual2

RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

RUN apt-get install python3 python3-pip -y
# TODO(AMJ): Get this to compile, determine base & pip requirementes
RUN pip3 install absl-py

COPY staging/ /app
WORKDIR /app

COPY evaluator_ringmaster_wrapper.py /app

# long series of args here.
CMD ["python3", "evaluator_ringmaster_wrapper.py"]
