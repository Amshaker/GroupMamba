#!/usr/bin/env sh

# Set environment variables
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29600

# Model and training settings
MODEL="groupmamba_tiny"
BATCH_SIZE=128
DATA_PATH="ILSVRC2012_imgs"
OUTPUT_DIR="GroupMamba_tiny_output"

# Launch distributed training
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --groupmamba-model $MODEL \
    --batch-size $BATCH_SIZE \
    --data-path $DATA_PATH \
    --output $OUTPUT_DIR \
