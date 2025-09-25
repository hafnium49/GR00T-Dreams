#!/bin/bash
# paper_return.sh - Preprocessing script for Hafnium49/paper_return dataset

echo "Starting preprocessing for paper_return dataset..."

# Set dataset paths
DATASET_PATH="./paper_return_dataset"
OUTPUT_PATH="./paper_return_processed"

# Step 1: Download dataset (if not already downloaded)
if [ ! -d "$DATASET_PATH" ]; then
    echo "Downloading paper_return dataset..."
    huggingface-cli download Hafnium49/paper_return --local-dir "$DATASET_PATH"
fi

# Step 2: Validate dataset structure
echo "Validating dataset structure..."
conda run -p /home/hafnium/GR00T-Dreams/.conda python -c "
import json
from pathlib import Path
import sys

dataset_path = Path('$DATASET_PATH')

# Check required files
required_files = [
    'meta/episodes.jsonl',
    'meta/info.json', 
    'meta/tasks.jsonl',
    'meta/modality.json'
]

missing_files = []
for file_path in required_files:
    full_path = dataset_path / file_path
    if not full_path.exists():
        missing_files.append(file_path)
        
if missing_files:
    print(f'❌ Missing required files: {missing_files}')
    print('Please ensure your dataset follows LeRobot v2.0 format')
    sys.exit(1)
else:
    print('✅ All required files found')
    
# Load and validate modality.json
with open(dataset_path / 'meta/modality.json') as f:
    modality = json.load(f)
    
print(f'Video keys: {list(modality.get(\"video\", {}).keys())}')
print(f'State keys: {list(modality.get(\"state\", {}).keys())}')
print(f'Action keys: {list(modality.get(\"action\", {}).keys())}')
"

if [ $? -ne 0 ]; then
    echo "Dataset validation failed. Exiting."
    exit 1
fi

# Step 3: Test dataset loading with GR00T
echo "Testing dataset loading with GR00T..."
conda run -p /home/hafnium/GR00T-Dreams/.conda python -c "
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
import sys

try:
    # Load dataset
    dataset = LeRobotSingleDataset(
        path='$DATASET_PATH',
        embodiment_tag=EmbodimentTag.SO100,
    )
    
    print(f'✅ Dataset loaded successfully!')
    print(f'Number of episodes: {len(dataset)}')
    
    # Check sample data
    sample = dataset[0]
    print(f'Sample data keys: {list(sample.keys())}')
    
    # Check dimensions
    for key in ['observation.state', 'action']:
        if key in sample:
            print(f'{key} shape: {sample[key].shape}')
    
    # Check video keys
    video_keys = [k for k in sample.keys() if 'observation.images' in k]
    for key in video_keys:
        print(f'{key} shape: {sample[key].shape}')
        
except Exception as e:
    print(f'❌ Dataset loading failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Dataset loading test failed. Please check your dataset format."
    exit 1
fi

echo "✅ Dataset preprocessing completed successfully!"
echo "Your dataset is ready for training with GR00T Dreams."
echo ""
echo "Next steps:"
echo "1. Run fine-tuning: PYTHONPATH=. torchrun scripts/gr00t_finetune.py --dataset-path \"$DATASET_PATH\" --data-config so100 --embodiment_tag \"so100\""
echo "2. Or use IDM pipeline for advanced preprocessing (see full guide)"