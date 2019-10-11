cd $(pwd)
export MXNET_CPU_WORKER_NTHREADS=64; \
#export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice;
CUDA_VISIBLE_DEVICES='0,1,2,3' \
    python3 train_parall.py \
        --network r100 \
        --loss arcface \
        --dataset train_1m \
		--per-batch-size 224
