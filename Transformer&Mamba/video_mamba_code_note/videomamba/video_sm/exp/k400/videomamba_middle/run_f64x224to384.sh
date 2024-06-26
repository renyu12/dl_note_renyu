export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_middle_f64_res224to384'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='your_k400_path'
DATA_PATH='your_k400_metadata_path'

PARTITION='video5'
GPUS=16
GPUS_PER_NODE=8
CPUS_PER_TASK=16

# renyu: middle 64帧 384*384升分辨率模型是middle 64帧 224*224模型微调来的
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        python run_class_finetuning.py \
        --model videomamba_middle \
        --finetune your_model_path/videomamba_m16_k400_f64_res224.pth \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --update_freq 2 \
        --num_sample 2 \
        --input_size 384 \
        --short_side_size 384 \
        --save_ckpt_freq 100 \
        --num_frames 64 \
        --orig_t_size 64 \
        --num_workers 12 \
        --warmup_epochs 1 \
        --tubelet_size 1 \
        --epochs 6 \
        --lr 5e-6 \
        --drop_path 0.8 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 1e-8 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16
