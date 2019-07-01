#!/bin/bash

set -euo pipefail

if [[ $# -eq 0 ]] ; then
    echo 'Usage: build.sh dst_dir'
    exit 1
fi

src_dir=${BASH_SOURCE[0]}.runfiles
dst_dir=$1

echo "Copying from \"${src_dir}\" to \"${dst_dir}\""

rm -rfd ${dst_dir}/*/

rsync -a --copy-links ${src_dir}/__main__/cc/tensorflow/*.so ${dst_dir}/lib/
rsync -a --copy-links ${src_dir}/org_tensorflow/tensorflow/*.so ${dst_dir}/lib/
rsync -a --copy-links --exclude "*.so" ${src_dir}/org_tensorflow/ ${dst_dir}/include/
rsync -a --copy-links ${src_dir}/eigen_archive/ ${dst_dir}/include/third_party/eigen3/
rsync -a --copy-links ${src_dir}/protobuf_archive/src/ ${dst_dir}/include/
