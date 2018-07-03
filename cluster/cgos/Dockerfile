FROM gcr.io/${PROJECT}/cc-base:0.17

RUN apt-get update && apt-get install gettext -y

COPY cgosGtp-linux-x86_64 /app

ARG MODEL
ARG MODEL_NUM

ENV MODEL=$MODEL
ENV MODEL_NUM=$MODEL_NUM

WORKDIR /app

RUN gsutil cp gs://minigo-pub/v7-19x19/models/$MODEL /app
COPY config.txt /app
RUN envsubst < config.txt > config.txt

RUN cat config.txt
CMD ["./cgosGtp-linux-x86_64", "-c", "config.txt"]
