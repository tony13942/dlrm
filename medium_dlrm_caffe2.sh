#!/bin/bash

# the following variable embedding_size evaluates to 10000000-10000000-10000000....8 times.
# where 1000000 is the average number of rows per table and 64 is the number of embedding tables for medium config
embedding_size=$(printf '10000000-%.0s' {1..7})
embedding_size+=10000000

sparse_feature_size=64

python dlrm_s_caffe2.py \
--arch-sparse-feature-size $sparse_feature_size \
--arch-embedding-size $embedding_size \
--arch-mlp-bot "1600-1024-1024-1024-1024-$sparse_feature_size" \
--arch-mlp-top "2048-2048-2048-2048-2048-2048-2048-2048-2048-1" \
--arch-interaction-op "dot" \
--activation-function "relu" \
--num-batches 1000 \
--num-indices-per-lookup 60 \
--mini-batch-size 512 \
--nepochs 1 \
--caffe2-net-type "async_dag" \
--use-gpu \
--print-time
