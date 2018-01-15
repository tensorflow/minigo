FROM ubuntu:16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -f -y pkg-config python3 python3-pip libffi6 libfreetype6 libopenblas-base libffi-dev libfreetype6-dev libpng-dev libopenblas-dev curl

ADD staging/requirements-cpu.txt /app/requirements-cpu.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements-cpu.txt

ADD staging/ /app

WORKDIR /app

ENTRYPOINT ["./player_wrapper.sh"]
