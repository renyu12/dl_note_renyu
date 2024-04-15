export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_middle_f16_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='your_k400_path'
DATA_PATH='your_k400_metadata_path'

PARTITION='video5'
GPUS=16
GPUS_PER_NODE=8
CPUS_PER_TASK=16

# renyu: middle 16帧 224*224模型，2*8gpu训练
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        python run_class_finetuning.py \    # renyu: 各个从头训练的脚本都是通过run_class_finetuning.py脚本启动的
        --model videomamba_middle \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 16 \
        --num_workers 12 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 50 \
        --lr 1e-4 \
        --drop_path 0.8 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16
