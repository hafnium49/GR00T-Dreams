# GR00T Dreams Dataset Synthesis Guide: Hafnium49/paper_return_front_view Dataset

This guide will help you synthesize an augmented dataset using your `Hafnium49/paper_return_front_view` dataset with the GR00T Dreams framework for SO-101 robotic arm training. The goal is to generate synthetic training data that can be used to train robust policies for SO-101. This dataset contains two camera views, but we'll use only the main camera (ignoring secondary_0).

## Dataset Analysis

Based on the Hugging Face metadata, your dataset:
- ✅ Is **SO-100 compatible** (tagged with `so100`) - Perfect base for SO-101 synthesis
- ✅ Is **LeRobot compatible** (tagged with robotics task categories)
- ✅ Contains **multi-modal data** (tabular + video) - Ideal for Cosmos fine-tuning
- ✅ Uses **parquet format** for structured data
- ✅ Was generated with **phosphobot** framework
- 🎯 **Target**: Generate synthetic variations for SO-101 training

## Prerequisites

1. **Download your dataset** (requires HF authentication if private):
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the dataset
huggingface-cli login  # Login if dataset is private
huggingface-cli download Hafnium49/paper_return_front_view --local-dir ./paper_return_front_view_dataset
```

2. **Verify dataset structure**:
Your dataset should follow this LeRobot-compatible structure:
```
paper_return_dataset/
├── meta/
│   ├── episodes.jsonl
│   ├── info.json
│   ├── modality.json     # ← Key file for GR00T compatibility
│   └── tasks.jsonl
├── videos/
│   └── chunk-000/
│       └── observation.images.{camera_name}/
│           ├── episode_000000.mp4
│           └── episode_000001.mp4
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        └── episode_000001.parquet
```

## Step 1: Verify/Create Modality Configuration

Since your dataset is SO-100 compatible, you likely already have the correct `meta/modality.json`. If not, create one based on the SO-100 template:

```json
{
    "state": {
        "main_shoulder_pan": {
            "start": 0,
            "end": 1
        },
        "main_shoulder_lift": {
            "start": 1,
            "end": 2
        },
        "main_elbow_flex": {
            "start": 2,
            "end": 3
        },
        "main_wrist_flex": {
            "start": 3,
            "end": 4
        },
        "main_wrist_roll": {
            "start": 4,
            "end": 5
        },
        "main_gripper": {
            "start": 5,
            "end": 6
        }
    },
    "action": {
        "main_shoulder_pan": {
            "start": 0,
            "end": 1,
            "absolute": false
        },
        "main_shoulder_lift": {
            "start": 1,
            "end": 2,
            "absolute": false
        },
        "main_elbow_flex": {
            "start": 2,
            "end": 3,
            "absolute": false
        },
        "main_wrist_flex": {
            "start": 3,
            "end": 4,
            "absolute": false
        },
        "main_wrist_roll": {
            "start": 4,
            "end": 5,
            "absolute": false
        },
        "main_gripper": {
            "start": 5,
            "end": 6,
            "absolute": false
        }
    },
    "video": {
        "webcam": {
            "original_key": "observation.images.webcam"
        }
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}
```

**Note**: Adjust the camera names and any additional cameras if your setup is different.

## Step 2: Load and Validate Dataset

Create a simple test script to verify your dataset loads correctly:

```python
# test_dataset_loading.py
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag

# Load your dataset
dataset = LeRobotSingleDataset(
    path="./paper_return_dataset",
    embodiment_tag=EmbodimentTag.SO100,  # Use SO100 since your dataset is compatible
)

print(f"Dataset loaded successfully!")
print(f"Number of episodes: {len(dataset)}")
print(f"Sample data keys: {list(dataset[0].keys())}")

# Check video and state dimensions
sample = dataset[0]
if 'observation.images.webcam' in sample:
    print(f"Video shape: {sample['observation.images.webcam'].shape}")
if 'observation.state' in sample:
    print(f"State shape: {sample['observation.state'].shape}")
if 'action' in sample:
    print(f"Action shape: {sample['action'].shape}")
```

Run the test:
```bash
conda run -p /home/hafnium/GR00T-Dreams/.conda python test_dataset_loading.py
```

## Step 3: Cosmos Fine-tuning for Dataset Synthesis

The main workflow is to fine-tune Cosmos video world model on your dataset, then generate synthetic variations. This creates the training data for SO-101 adaptation.

### Step 3.1: Prepare Data for Cosmos Fine-tuning

Convert your filtered dataset for Cosmos post-training:

```bash
# Install cosmos-predict2 environment first
# See: https://github.com/nvidia-cosmos/cosmos-predict2

# Convert to Cosmos format
python IDM_dump/raw_to_lerobot.py \
    --input_dir "./paper_return_filtered_dataset" \
    --output_dir "./paper_return_cosmos_ready" \
    --cosmos_predict2

# Generate text prompts for training
# Create prompts like "Move white paper into red square on table"
python scripts/generate_cosmos_prompts.py \
    --dataset_path "./paper_return_cosmos_ready" \
    --task_description "paper_manipulation"
```

### Step 3.2: Fine-tune Cosmos Video2World Model

Post-train Cosmos on your paper manipulation patterns:

```bash
# In cosmos-predict2 repository
torchrun --nproc_per_node=2 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  -- experiment="predict2_video2world_training_2b_paper_return"
```

### Step 3.3: Generate Synthetic Dream Videos

Create hundreds of variations using your fine-tuned model:

```bash
# Generate dreams with varied conditions
for i in {1..100}; do
  python examples/video2world.py \
    --model_size 2B \
    --dit_path checkpoints/paper_return_cosmos/model.pt \
    --prompt "Move white paper into red square with varied lighting" \
    --input_path seeds/seed_frame_$i.jpg \
    --save_path results/paper_dreams/dream_$i.mp4
done
```

## Step 4: Extract Actions from Synthetic Dreams

Convert synthetic videos back to actionable training data using IDM:

### Step 4.1: Prepare Dream Videos for IDM

```bash
# Convert dream outputs to IDM format
python IDM_dump/convert_directory.py \
    --input_dir "results/paper_dreams" \
    --output_dir "results/paper_dreams_idm_ready"
```

### Step 4.2: Extract Actions using SO-100 IDM

```bash
# Use existing SO-100 IDM model (or train custom SO-101 IDM)
python IDM_dump/dump_idm_actions.py \
    --checkpoint "seonghyeonye/IDM_so100" \
    --dataset "results/paper_dreams_idm_ready" \
    --output_dir "./paper_dreams_with_actions" \
    --num_gpus 1 \
    --video_indices "0 8"
```

### Step 4.3: Convert to LeRobot Format

```bash
# Convert IDM outputs to LeRobot episodes
python IDM_dump/idm_to_lerobot.py \
    --input_dir "./paper_dreams_with_actions" \
    --output_dir "./synthetic_paper_episodes" \
    --embodiment_tag "so101"
```

### Step 4.4: Custom SO-101 Embodiment (Optional)

If SO-101 differs significantly from SO-100, create custom configuration:

```python
# In gr00t/data/embodiment_tags.py
SO101 = "so101"

# Add modality.json for SO-101 in IDM_dump/global_metadata/so101/
# Add data config in gr00t/experiment/data_config_idm.py
```

## Step 5: Create Augmented Dataset for SO-101 Training

### Step 5.1: Merge Real and Synthetic Data

```bash
# Combine original dataset with synthetic episodes
python scripts/merge_datasets.py \
    --real_dataset "./paper_return_filtered_dataset" \
    --synthetic_dataset "./synthetic_paper_episodes" \
    --output_dataset "./paper_return_augmented_so101" \
    --merge_ratio 0.7  # 70% real, 30% synthetic
```

### Step 5.2: Validate Augmented Dataset

```bash
# Test the merged dataset
python paper_return_examples.py ./paper_return_augmented_so101

# Check episode count and quality
echo "Original episodes: $(cat paper_return_filtered_dataset/meta/episodes.jsonl | wc -l)"
echo "Synthetic episodes: $(cat synthetic_paper_episodes/meta/episodes.jsonl | wc -l)"
echo "Total episodes: $(cat paper_return_augmented_so101/meta/episodes.jsonl | wc -l)"
```

### Step 5.3: Export for GR00T N1.5 Training

Your augmented dataset is now ready for training SO-101 policies:

```bash
# Example training command (to be used with GR00T N1.5)
PYTHONPATH=. torchrun scripts/gr00t_finetune.py \
    --dataset-path "./paper_return_augmented_so101" \
    --data-config so101 \
    --embodiment_tag "so101" \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --wandb-project "so101_paper_return_training"
```

## Troubleshooting

### Common Issues:

1. **Dataset format issues**: Ensure your dataset follows the exact LeRobot v2.0 format
2. **Missing modality.json**: Create one based on the SO-100 template above
3. **Video codec issues**: Ensure videos are in H.264 format
4. **State/action dimension mismatch**: Verify the dimensions in your modality.json match your actual data

### Debug Script:
```python
# debug_dataset.py
import json
from pathlib import Path

dataset_path = Path("./paper_return_dataset")

# Check required files
required_files = [
    "meta/episodes.jsonl",
    "meta/info.json", 
    "meta/tasks.jsonl",
    "meta/modality.json"
]

for file_path in required_files:
    full_path = dataset_path / file_path
    if full_path.exists():
        print(f"✅ {file_path} exists")
        if file_path.endswith('.json'):
            with open(full_path) as f:
                data = json.load(f)
                print(f"   Keys: {list(data.keys())}")
    else:
        print(f"❌ {file_path} missing")

# Check data structure
data_dir = dataset_path / "data" / "chunk-000"
video_dir = dataset_path / "videos" / "chunk-000"

print(f"\nData files: {len(list(data_dir.glob('*.parquet')))} parquet files")
print(f"Video dirs: {list(video_dir.iterdir()) if video_dir.exists() else 'None'}")
```

## Next Steps

1. **Setup Cosmos Environment** - Install cosmos-predict2 and dependencies
2. **Fine-tune Cosmos** - Post-train on your filtered paper_return dataset  
3. **Generate Dreams** - Create hundreds of synthetic video variations
4. **Extract Actions** - Use IDM to convert videos to actionable data
5. **Merge Datasets** - Combine real and synthetic for augmented training corpus
6. **Train SO-101 Policy** - Use augmented dataset with GR00T N1.5
7. **Deploy and Test** - Evaluate on real SO-101 hardware

## Summary

This pipeline transforms your `Hafnium49/paper_return_front_view` dataset into a rich, augmented training corpus for SO-101. The GR00T Dreams approach generates synthetic variations that improve policy robustness across different lighting, backgrounds, and manipulation approaches - crucial for real-world deployment.

**Key Benefits:**
- 🎯 **SO-101 Adaptation**: Synthetic data helps bridge SO-100 → SO-101 gap
- 🌟 **Improved Robustness**: Varied synthetic scenes increase generalization  
- 📈 **Data Efficiency**: Multiply your 88 real episodes into hundreds of variations
- 🔄 **Iterative Improvement**: Easy to generate more data as needed