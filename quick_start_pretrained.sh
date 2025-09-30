#!/bin/bash

# Quick Start Script - Using Pre-trained Models for SO-100
# This script demonstrates how to use GR00T Dreams WITHOUT fine-tuning video world models

echo "🤖 GR00T Dreams - Pre-trained Model Workflow"
echo "=============================================="
echo ""

# Activate environment
echo "📦 Step 1: Activate your environment"
echo "source activate_groot.sh"
echo ""

# Download your dataset  
echo "📊 Step 2: Download your paper_return dataset"
echo "huggingface-cli download Hafnium49/paper_return --local-dir ./paper_return_dataset"
echo ""

# Validate dataset format
echo "🔍 Step 3: Validate dataset compatibility"
echo "python paper_return_examples.py ./paper_return_dataset"
echo ""

# Option 1: Skip to IDM action extraction (recommended)
echo "🎯 RECOMMENDED PATH: Skip video world model training"
echo ""
echo "📹 Step 4a: Use pre-trained Cosmos models for synthetic video generation"
echo "# This step would use pre-trained models - see cosmos-predict2 inference examples"
echo "# OR skip synthetic data generation and work with real data only"
echo ""

echo "🎬 Step 4b: Extract actions using existing SO-100 IDM"
echo "bash IDM_dump/scripts/preprocess/so100.sh"
echo ""

echo "🚀 Step 4c: Fine-tune GR00T N1 policy on your data"
echo "bash IDM_dump/scripts/finetune/so100.sh"
echo ""

# Option 2: Custom training path
echo "⚙️  ADVANCED PATH: Custom video world model training"
echo "# Only do this if you need robot-specific improvements"
echo "# Requires 8+ GPUs and significant computational resources"
echo "./cosmos_predict2_training_guide.sh"
echo ""

echo "💡 RECOMMENDATION:"
echo "   Start with the pre-trained models and real data training"
echo "   Add synthetic data generation later if needed for performance"
echo ""

echo "🎉 Your paper_return dataset is ready to use with SO-100 configuration!"