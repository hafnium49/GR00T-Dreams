#!/bin/bash
# paper_return_finetune.sh - Fine-tuning script for Hafnium49/paper_return dataset

echo "Starting GR00T N1 fine-tuning on paper_return dataset..."

# Configuration
DATASET_PATH="./paper_return_front_view_dataset"
OUTPUT_DIR="./paper_return_checkpoints"
BATCH_SIZE=4  # Adjust based on your GPU memory
LEARNING_RATE=1e-4
NUM_EPOCHS=100
WANDB_PROJECT="paper_return_groot"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Dataset not found at $DATASET_PATH"
    echo "Please run the preprocessing script first:"
    echo "bash IDM_dump/scripts/preprocess/paper_return.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Training configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR" 
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  W&B project: $WANDB_PROJECT"
echo ""

# Run fine-tuning
PYTHONPATH=. torchrun scripts/gr00t_finetune.py \
    --dataset-path "$DATASET_PATH" \
    --data-config so100 \
    --embodiment_tag "so100" \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --output-dir "$OUTPUT_DIR" \
    --wandb-project "$WANDB_PROJECT"

if [ $? -eq 0 ]; then
    echo "✅ Fine-tuning completed successfully!"
    echo "Checkpoints saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate your model using getting_started/1_gr00t_inference.ipynb"
    echo "2. Deploy your policy using getting_started/5_policy_deployment.md"
else
    echo "❌ Fine-tuning failed. Please check the error messages above."
    exit 1
fi