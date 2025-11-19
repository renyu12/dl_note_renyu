export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='videomamba_small_res224'
OUTPUT_DIR="$(dirname $0)"
LOG_DIR="./logs/${JOB_NAME}"
PARTITION='video5'
NNODE=1
NUM_GPUS=8
NUM_CPU=128

# renyu: 训练最核心的small 224*224模型，用的8卡服务器srun提交的
# renyu: srun命令是slurm作业调度器的作业提交命令
srun --mpi=pmi2 \                   # renyu: slurm的Message Passing Interface用pmi2库
    -p ${PARTITION} \               # renyu: 任务在哪个分区上跑（slurm中将节点划分为多个分区）
    -n${NNODE} \                    # renyu: 节点数（服务器数量）
    --gres=gpu:${NUM_GPUS} \        # renyu: 每个节点GPU数
    --ntasks-per-node=1 \           # renyu: 每个节点任务数（TODO: 是因为只有1个任务吗？不然应该分8个=GPU数量吧？）
    --cpus-per-task=${NUM_CPU} \    # renyu: 每个任务的CPU数（TODO: 全分完了，是因为只有一个任务吗？）
    python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env main.py \    # renyu: 把launch库当做脚本执行，应该是方便分布式训练的
        --root_dir_train your_imagenet_path/train/ \
        --meta_file_train your_imagenet_path/meta/train.txt \
        --root_dir_val your_imagenet_path/val/ \
        --meta_file_val your_imagenet_path/meta/val.txt \
        --model videomamba_small \
        --batch-size 512 \
        --num_workers 16 \
        --lr 5e-4 \
        --weight-decay 0.05 \
        --drop-path 0.15 \
        --no-model-ema \
        --output_dir ${OUTPUT_DIR}/ckpt \
        --bf16 \
        --dist-eval