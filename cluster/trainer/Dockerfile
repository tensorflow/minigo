ARG PROJECT
FROM gcr.io/$PROJECT/cc-base:latest

RUN pip3 install tensorflow==1.12
WORKDIR /app

ENV BOARD_SIZE="19"

COPY staging /app
COPY staging/rl_loop/ /app
COPY staging/mask_flags.py /app

CMD ["sh", "-c", "python rl_loop/train_and_validate.py --bucket_name=$BUCKET_NAME --pro_dataset=$PRO_DATASET"]
