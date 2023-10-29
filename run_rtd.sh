#!/bin/bash

# Set the paths to the generator and discriminator files
GENERATOR_CONFIG="deberta-v3-xsmall-changed/generator_config.json"
GENERATOR_WEIGHTS="deberta-v3-xsmall-changed/generator_3_epochs.bin"
DISCRIMINATOR_CONFIG="deberta-v3-xsmall-changed/config.json"
DISCRIMINATOR_WEIGHTS="deberta-v3-xsmall-changed/discriminator_3_epochs.bin"

# Run arguments
LOG_WITH="tensorboard"
PROJECT_DIR="debertinha-v2-accelerate"
RUN_NAME="debertinha-v2-runs"

# Dataset
DATASET_PATHS="ds_subset_encoded"

# Set the training arguments
TOKENIZER_NAME="deberta-v3-xsmall-changed"
BATCH_SIZE=1
TEMPERATURE=1.0
RTD_LAMBDA=20.0
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.0
MIXED_PRECISION="no"
NUM_WARMUP_STEPS=10000
GRADIENT_ACCUMULATION_STEPS=2
LR_SCHEDULER_TYPE="linear"
NUM_TRAIN_EPOCHS=1
CPU=false
CHECKPOINTING_STEPS=10
SAVE_TOTAL_LIMIT=1
MAX_GRAD_NORM=1.0

# Run the script with the specified arguments
accelerate launch run_rtd.py \
    --generator_config $GENERATOR_CONFIG \
    --generator_weights $GENERATOR_WEIGHTS \
    --discriminator_config $DISCRIMINATOR_CONFIG \
    --discriminator_weights $DISCRIMINATOR_WEIGHTS \
    --per_device_train_batch_size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --rtd_lambda $RTD_LAMBDA \
    --tokenizer_name $TOKENIZER_NAME \
    --learning_rate $LEARNING_RATE \
    --mixed_precision $MIXED_PRECISION \
    --weight_decay $WEIGHT_DECAY \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_warmup_steps $NUM_WARMUP_STEPS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --cpu $CPU \
    --log_with $LOG_WITH \
    --project_dir $PROJECT_DIR \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --max_grad_norm $MAX_GRAD_NORM \
    --dataset_paths $DATASET_PATHS \
    --run_name $RUN_NAME