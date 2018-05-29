#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst_dir="${script_dir}/tensorflow"
tmp_dir="/tmp/minigo_tf"
tmp_pkg_dir="/tmp/tensorflow_pkg"
rm -rfd ${tmp_dir}
rm -rfd ${tmp_pkg_dir}
mkdir -p ${tmp_dir}

# The TensorFlow 1.8.0 release doesn't compile with gcc6+, so checkout at the
# commit that fixed the build issue.
# See https://github.com/tensorflow/tensorflow/issues/18402 for more details.
# TODO(tommadams): switch to v.18.1 when that's released.
# TODO(tommadams): we should probably switch to Clang at some point.
commit_tag="e489b600f388ae345387881a85368af3cd373ba2"

echo "Cloning tensorflow to ${tmp_dir}"
git clone https://github.com/tensorflow/tensorflow "${tmp_dir}"

pushd "${tmp_dir}"

echo "Checking out ${commit_tag}"
git checkout "${commit_tag}"

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"
CC_OPT_FLAGS="-march=ivybridge" \
TF_NEED_JEMALLOC=1 \
TF_NEED_GCP=1 \
TF_NEED_HDFS=0 \
TF_NEED_S3=0 \
TF_NEED_KAFKA=0 \
TF_NEED_CUDA=1 \
TF_NEED_GDR=0 \
TF_NEED_VERBS=0 \
TF_NEED_OPENCL_SYCL=0 \
TF_CUDA_CLANG=0 \
TF_NEED_TENSORRT=0 \
TF_NEED_MPI=0 \
TF_SET_ANDROID_WORKSPACE=0 \
./configure

echo "Building tensorflow package"
bazel build --copt=-march=ivybridge -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${tmp_pkg_dir}

echo "Tensorflow built-ish"
echo "Unpacking tensorflow package..."
unzip -q ${tmp_pkg_dir}/tensorflow-*.whl -d ${tmp_dir}

echo "Copying tensor flow headers to ${dst_dir}"
cp -r ${tmp_dir}/tensorflow-*.data/purelib/tensorflow/include "${dst_dir}"

echo "Building tensorflow libraries"
bazel build --copt=-march=ivybridge -c opt --config=opt //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so

echo "Copying tensorflow libraries to ${dst_dir}"
cp bazel-bin/tensorflow/libtensorflow_*.so "${dst_dir}"

echo "Building toco"
bazel build -c opt --config=opt --copt=-march=ivybridge //tensorflow/contrib/lite/toco:toco
cp bazel-bin/tensorflow/contrib/lite/toco/toco "${dst_dir}"

popd
echo "Deleting tmp dir ${tmp_dir}"
#rm -rf "${tmp_dir}"
