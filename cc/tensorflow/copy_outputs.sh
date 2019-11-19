#!/bin/bash

set -euo pipefail

if [[ $# -eq 0 ]] ; then
    echo 'Usage: build.sh dst_dir'
    exit 1
fi

src_dir=${BASH_SOURCE[0]}.runfiles
dst_dir=$1

echo "Copying from \"${src_dir}\" to \"${dst_dir}\""

for sub_dir in lib include bin; do
  rm -rfd "${dst_dir}/${sub_dir}"
  mkdir -p "${dst_dir}/${sub_dir}"
done

rsync -a --copy-links ${src_dir}/__main__/cc/tensorflow/*.so ${dst_dir}/lib/
rsync -a --copy-links ${src_dir}/org_tensorflow/tensorflow/*.so.1.15.0 ${dst_dir}/lib/
rsync -a --copy-links --exclude "*.so.1.15.0" ${src_dir}/org_tensorflow/ ${dst_dir}/include/
rsync -a --copy-links ${src_dir}/eigen_archive/ ${dst_dir}/include/third_party/eigen3/
rsync -a --copy-links ${src_dir}/com_google_protobuf/src/ ${dst_dir}/include/
rsync -a --copy-links ${src_dir}/org_tensorflow/tensorflow/lite/toco/toco ${dst_dir}/bin/

mv ${dst_dir}/lib/libtensorflow_cc.so.1{.15.0,}
mv ${dst_dir}/lib/libtensorflow_framework.so.1{.15.0,}

