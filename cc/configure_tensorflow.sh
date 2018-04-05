#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst_dir="${script_dir}/tensorflow"
tmp_dir="$(mktemp -d -p /tmp minigo_tf.XXXXXX)"
version_tag="v1.6.0-rc1"

echo "Cloning tensorflow to ${tmp_dir}"
git clone https://github.com/tensorflow/tensorflow "${tmp_dir}"

pushd "${tmp_dir}"

echo "Checking out ${version_tag}"
git checkout "tags/${version_tag}"

echo "Configuring tensorflow"
./configure

echo "Building tensorflow package"
bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

echo "Unpacking tensorflow package"
wheel unpack /tmp/tensorflow_pkg/tensorflow-*.whl

echo "Copying tensor flow headers to ${dst_dir}"
cp -r tensorflow-*/tensorflow-*.data/purelib/tensorflow/include "${dst_dir}"

echo "Building tensorflow libraries"
bazel build -c opt --config=opt //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so

echo "Copying tensorflow libraries and headers to ${dst_dir}"
cp bazel-bin/tensorflow/libtensorflow_*.so "${dst_dir}"

echo "Deleting tmp dir ${tmp_dir}"
rm -rf "${tmp_dir}"
popd
