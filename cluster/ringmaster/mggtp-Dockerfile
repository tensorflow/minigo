ARG PROJECT
FROM gcr.io/$PROJECT/cc-base:latest

COPY staging/ /app
RUN bazel build -c opt --define=tf=1 --define=gpu=1 --define=bt=1 cc/gtp

ENTRYPOINT ["bazel-bin/cc/gtp"]
