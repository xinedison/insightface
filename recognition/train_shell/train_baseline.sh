cd $(pwd)
CUDA_VISIBLE_DEVICES='0,1' \
    python train_parall.py \
        --network r100 \
        --loss arcface \
        --dataset emore
