# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Docker image test-harness for running python tests

FROM python:3.6-slim

# Never prompt the user for choices on installation/configuration of packages
ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

# Define en_US.
ENV LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8 \
    LC_MESSAGES=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# buildDeps should be packages needed only to build some other packages as
# these packages are purged in a later step.
#
# gcc & python-dev are needed so we can install crcmod for gsutil
RUN set -ex  \
    && apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends \
        git \
        curl \
        locales \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

# Set the locale
RUN sed -i 's/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

COPY bootstrap.sh /usr/local/bin
RUN chmod a+x /usr/local/bin/bootstrap.sh

COPY ../requirements-cpu.txt /usr/local/requirements-cpu.txt
RUN chmod a+r /usr/local/requirements-cpu.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /usr/local/requirements-cpu.txt

ENTRYPOINT ["/usr/local/bin/bootstrap.sh"]
