#!/bin/bash

# Cosmos-Predict2 Setup Guide for Video World Model Training
# This script outlines the setup process for cosmos-predict2 training environment

echo "🚨 IMPORTANT: This requires significant computational resources!"
echo "   - Minimum: 8 high-end GPUs (A100, H100)"
echo "   - Recommended: 32 GPUs across 4 nodes for 14B model"
echo "   - Storage: 500GB+ available space"
echo "   - Time: Days to weeks for training"
echo ""

# Step 1: Clone cosmos-predict2 repository
echo "📁 Step 1: Setting up cosmos-predict2 repository"
echo "git clone https://github.com/nvidia-cosmos/cosmos-predict2.git"
echo "cd cosmos-predict2"
echo ""

# Step 2: Environment setup
echo "🐍 Step 2: Environment setup"
echo "# Follow cosmos-predict2 setup guide:"
echo "# https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/setup.md"
echo ""

# Step 3: Download model checkpoints
echo "📦 Step 3: Download base model checkpoints"
echo "# Download Cosmos-Predict2 base models following the setup guide"
echo ""

# Step 4: Prepare training data
echo "📊 Step 4: Download GR1 training dataset (~100GB)"
echo "huggingface-cli download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/"
echo "mkdir -p datasets/benchmark_train/gr1/videos"
echo "mv datasets/benchmark_train/hf_gr1/gr1/*mp4 datasets/benchmark_train/gr1/videos"
echo "mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/"
echo ""

# Step 5: Preprocess data
echo "🔄 Step 5: Preprocess training data"
echo "python -m scripts.get_t5_embeddings_from_groot_dataset --dataset_path datasets/benchmark_train/gr1"
echo ""

# Step 6: Training commands
echo "🚀 Step 6: Training commands"
echo ""
echo "# For 2B model (8 GPUs):"
echo "EXP=predict2_video2world_training_2b_groot_gr1_480"
echo "torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=\${EXP}"
echo ""
echo "# For 14B model (4 nodes x 8 GPUs = 32 GPUs):"
echo "EXP=predict2_video2world_training_14b_groot_gr1_480"
echo "NVTE_FUSED_ATTN=0 torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint \$MASTER_ADDR:1234 \\"
echo "  -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=\${EXP}"
echo ""

echo "⚠️  NOTE: This is a separate training pipeline from GR00T Dreams"
echo "   The trained models will be used BY GR00T Dreams for synthetic data generation"
echo "   Consider using pre-trained models unless you have specific customization needs"