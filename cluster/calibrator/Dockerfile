ARG PROJECT
FROM gcr.io/$PROJECT/cc-base:latest

RUN pip3 install tensorflow==1.13.1
WORKDIR /app

ENV BOARD_SIZE="19"

COPY staging /app
COPY staging/rl_loop/ /app

CMD ["sh", "-c", "python rl_loop/update_resign_threshold.py --bucket_name=$BUCKET_NAME --flagfile=rl_loop/distributed_flags"]
