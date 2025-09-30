# GR00T Dreams Workflow - Using Pre-trained Models
# Recommended approach for SO-100 and paper_return dataset

## Option A: Skip Video World Model Fine-tuning (Recommended)
# Use pre-trained Cosmos models directly

## Step 1: Generate Synthetic Videos (Using Pre-trained Models)
# Use pre-trained Cosmos-Predict2 models to generate synthetic robot videos
# Input: Your paper_return dataset images + text prompts
# Output: Synthetic robot videos

## Step 2: Extract IDM Actions  
# Use the existing SO-100 IDM model to extract actions from synthetic videos
bash IDM_dump/scripts/preprocess/so100.sh

## Step 3: Fine-tune GR00T N1
# Train the robot policy on your real + synthetic data
bash IDM_dump/scripts/finetune/so100.sh

## Option B: Custom Video World Model Fine-tuning (Advanced)
# Only if you need robot-specific video generation improvements

### Step 1a: Fine-tune Video World Models (Optional)
# Follow cosmos-predict2 walkthrough to fine-tune on your robot data
# Requires 8+ GPUs and significant time

### Step 1b: Generate Synthetic Videos (Using Your Fine-tuned Model) 
# Use your custom model for better robot-specific generation

### Steps 2-3: Same as Option A

## 🎯 For Your Use Case (SO-100 + paper_return):
# Recommendation: Start with Option A (pre-trained models)
# You can always fine-tune later if you need improvements