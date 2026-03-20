export PROJECT_NAME=string-task
export EXPERIMENT_NAME=stage2-rft-rl-checkpoint
export MODEL_PATH=/data/user_data/gyeongwk/checkpoints/string-task/stage1-rft
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=data/string_task/stage2_rft_level2-rl_checkpoint/rft_data/train.parquet 
export VAL_FILES=data/string_task/stage2_rft_level2-rl_checkpoint/rft_data/test.parquet 
export BATCH_SIZE=128
export EPOCHS=1
export SAVE_DIR=/data/user_data/gyeongwk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh
