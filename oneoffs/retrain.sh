#!/bin/bash

# This script must be run from a Cloud TPU virtual machine.

# Retrains a new model from the v13 golden chunks using a Cloud TPU.
# See the "Retraining a model" section of the in README.md in the repository
# root for information on how to use this script.

function train() {
  trunk_layers=19
  filter_width=128
  tpu_name=$(hostname -s)
  lr_1=0.02
  lr_2=0.002
  lr_3=0.0002
  value_cost_weight=2
  gcs_bucket=$1

  # Train on the v13 model's data.
  all_chunks=(gs://minigo-pub/v13-19x19/data/golden_chunks/{1..700}.tfrecord.zz)
  num_all_chunks=${#all_chunks[@]}

  # Sample 1 in every 4 training chunks. This makes the model slightly weaker
  # because it's training on fewer examples, but training finishes 4x quicker.
  step=4

  i=0
  train_chunks=()
  while [[ $i -lt $(($num_all_chunks)) ]]; do
    train_chunks+=(${all_chunks[$i]})
    i=$(($i + $step))
  done

  num_train_chunks=${#train_chunks[@]}

  sub_dir="$(date +%Y-%m-%d-%M)"
  work_dir="gs://$gcs_bucket/train/$sub_dir"

  # Each chunk from v13 contains rouhly 2000000 examples.
  examples_per_chunk=2000000
  batch_size=64

  # The extra divide by 8 is because a TPU has 8 cores that each batch their
  # examples separately.
  steps_per_chunk=$((${examples_per_chunk} / ${batch_size} / 8))

  # Total number of steps we're expecting to take.
  num_steps=$((${num_train_chunks} * ${steps_per_chunk}))

  echo "Training on ${#train_chunks[@]} chunks in ${num_steps} steps"
  echo "Writing to $work_dir"

  echo gsutil cp "${BASH_SOURCE[0]}" "$work_dir/"

  echo python3 train.py \
    --trunk_layers=$trunk_layers \
    --conv_width=$filter_width \
    --fc_width=$filter_width \
    --use_tpu=true \
    --tpu_name=$tpu_name \
    --steps_to_train=$num_steps \
    --work_dir=$work_dir \
    --iterations_per_loop=256 \
    --train_batch_size=64 \
    --summary_steps=256 \
    --lr_boundaries=$((num_steps / 3)) \
    --lr_boundaries=$((num_steps * 2 / 3)) \
    --lr_rates=$lr_1 \
    --lr_rates=$lr_2 \
    --lr_rates=$lr_3 \
    --value_cost_weight=$value_cost_weight \
    ${train_chunks[@]}
}

train $1
