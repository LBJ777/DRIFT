
  # 实验 A：基线（天花板，用来展示方法的理论上限）
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  conda run --no-capture-output -n aigc python /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/experiments/phase1_binary.py \
    --data_dir /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/data/step_a \
    --train_generators ProGAN,SD_v1.5,Wukong \
    --test_generators  ProGAN,SD_v1.5,Wukong \
    --model_path /data/lizihao/AIGC/AIGCDetectBenchmark-main/weights/preprocessing/256x256_diffusion_uncond.pt \
    --output_dir ./results/phase1_baseline \
    --ddim_steps 20 --image_size 256 --epochs 10 \
     --precompute

  # 实验 B：核心实验（论文主表格）
  # 训练时 Wukong 不可见，测试时才出现
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  conda run --no-capture-output -n aigc python /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/experiments/phase1_binary.py \
    --data_dir /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/data/step_a \
    --train_generators ProGAN,SD_v1.5 \
    --test_generators  ProGAN,SD_v1.5,Wukong \
    --model_path /data/lizihao/AIGC/AIGCDetectBenchmark-main/weights/preprocessing/256x256_diffusion_uncond.pt \
    --output_dir ./results/phase1_crossgen \
    --ddim_steps 20 --image_size 256 --epochs 10 \
    --precompute

  # 实验 C：最强测试（硬泛化，bonus result）
  # 只用 GAN 训练，测试扩散模型
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  conda run --no-capture-output -n aigc python /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/experiments/phase1_binary.py \
    --data_dir /data/lizihao/AIGC/AIGCDetectBenchmark-main/DRIFT/data/step_a \
    --train_generators ProGAN \
    --test_generators  ProGAN,SD_v1.5,Wukong \
    --model_path /data/lizihao/AIGC/AIGCDetectBenchmark-main/weights/preprocessing/256x256_diffusion_uncond.pt \
    --output_dir ./results/phase1_hardcross \
    --ddim_steps 20 --image_size 256 --epochs 10 \
    --precompute