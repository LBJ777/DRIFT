cd /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT && \
mkdir -p results/step_a_real && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
# python -u experiments/step_a_validation.py \
#     --data_dir /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/data/step_a \
#     --model_path /data/lizihao/AIGC/AIGCDetectBenchmark-main/weights/preprocessing/256x256_diffusion_uncond.pt \
#     --output_dir ./results/step_a_real \
#     --num_samples 200 \
#     --ddim_steps 20 \
#     --image_size 256 2>&1 | tee results/step_a_real/run.log

python -u /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/experiments/step_a_validation_1.py \
    --data_dir /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/data/step_a \
    --model_path /data/lizihao/AIGC/AIGCDetectBenchmark-main/weights/preprocessing/256x256_diffusion_uncond.pt \
    --output_dir ./results/step_a_v2 \
    --num_samples 200 \
    --ddim_steps 20 \
    --image_size 256