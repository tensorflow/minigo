#!/bin/bash

set -e

# TODO(tommadams): we should probably switch to Clang at some point.

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"
CC_OPT_FLAGS="${CC_OPT_FLAGS:--march=native}" \
TF_NEED_JEMALLOC=${TF_NEED_JEMALLOC:-1} \
TF_NEED_GCP=${TF_NEED_GCP:-1} \
TF_NEED_HDFS=${TF_NEED_HDFS:-0} \
TF_NEED_S3=${TF_NEED_S3:-0} \
TF_NEED_KAFKA=${TF_NEED_KAFKA:-0} \
TF_NEED_GDR=${TF_NEED_GDR:-0} \
TF_NEED_VERBS=${TF_NEED_VERBS:-0} \
TF_NEED_OPENCL_SYCL=${TF_NEED_OPENCL_SYCL:-0} \
TF_CUDA_CLANG=${TF_CUDA_CLANG:-0} \
TF_NEED_MPI=${TF_NEED_MPI:-0} \
TF_SET_ANDROID_WORKSPACE=${TF_SET_ANDROID_WORKSPACE:-0} \
bazel --bazelrc=/dev/null run @org_tensorflow//:configure

output_base=$(bazel info output_base)
workspace=$(bazel info workspace)

# Copy TensorFlow's bazelrc files to workspace.
cp ${output_base}/external/org_tensorflow/tools/bazel.rc ${workspace}/tensorflow.bazelrc
cp ${output_base}/external/org_tensorflow/.tf_configure.bazelrc ${workspace}/tf_configure.bazelrc

echo
echo "You have the option to build TensorFlow once now, or you can build TensorFlow as a dependency
of MiniGo. You probably want the former unless you are planning to only run once or iterate on
changes to TensorFlow."

while true; do
    read -p "Would you like to build TensorFlow now (Y/n)? " -n 1 -r yn
    echo
    case ${yn:-Y} in
        [Yy]* )
            break;;
        [Nn]* )
            exit;;
    esac
done

echo "Building tensorflow package"
bazel run -c opt --config=opt //cc/tensorflow:build -- ${workspace}/cc/tensorflow

printf """
# This statement makes bazel use prebuilt binaries of TensorFlow.
build --action_env MG_PREBUILT_TF_PATH=cc/tensorflow
""" >> ${workspace}/tf_configure.bazelrc
