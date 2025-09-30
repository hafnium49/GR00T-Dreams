# GR00T Dreams Integration Guide: Hafnium49/paper_return_front_view Dataset

This guide will help you integrate your `Hafnium49/paper_return_front_view` dataset with the GR00T Dreams framework for SO-100 robotic arm training. This dataset contains two camera views, but we'll use only the main camera (ignoring secondary_0).

## Dataset Analysis

Based on the Hugging Face metadata, your dataset:
- ✅ Is **SO-100 compatible** (tagged with `so100`)
- ✅ Is **LeRobot compatible** (tagged with robotics task categories)
- ✅ Contains **multi-modal data** (tabular + video)
- ✅ Uses **parquet format** for structured data
- ✅ Was generated with **phosphobot** framework

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

## Step 3: Fine-tune GR00T N1 on Your Dataset

### Option A: Direct Fine-tuning (Recommended if data is already in good quality)

```bash
# Fine-tune directly on your paper_return dataset
PYTHONPATH=. torchrun scripts/gr00t_finetune.py \
    --dataset-path "./paper_return_dataset" \
    --data-config so100 \
    --embodiment_tag "so100" \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --num-epochs 100
```

### Option B: Use IDM Pipeline (if you want to extract additional synthetic actions)

If you want to use the full DreamGen pipeline with IDM action extraction:

1. **Preprocess your dataset** (adapt the SO-100 preprocessing script):
```bash
# Create a custom preprocessing script based on so100.sh
cp IDM_dump/scripts/preprocess/so100.sh IDM_dump/scripts/preprocess/paper_return.sh

# Edit the script to point to your dataset:
python IDM_dump/raw_to_lerobot.py \
    --input_dir "./paper_return_dataset" \
    --output_dir "./paper_return_processed" \
    --cosmos_predict2 

python IDM_dump/dump_idm_actions.py \
    --checkpoint "seonghyeonye/IDM_so100" \
    --dataset "./paper_return_processed" \
    --output_dir "./paper_return_idm" \
    --num_gpus 1 \
    --video_indices "0 8"
```

2. **Fine-tune on processed data**:
```bash
PYTHONPATH=. torchrun scripts/gr00t_finetune.py \
    --dataset-path "./paper_return_idm" \
    --data-config so100 \
    --embodiment_tag "so100"
```

## Step 4: Advanced Configuration (Optional)

### Multi-Camera Support
If your dataset has multiple cameras, update the `modality.json`:

```json
{
    "video": {
        "camera_1": {
            "original_key": "observation.images.camera_1"
        },
        "camera_2": {
            "original_key": "observation.images.camera_2"
        }
    }
}
```

And create a custom data config in `gr00t/experiment/data_config.py`:

```python
class PaperReturnDataConfig(BaseDataConfig):
    video_keys = ["video.camera_1", "video.camera_2"]  # Add all your cameras
    state_keys = ["state.main_shoulder_pan", "state.main_shoulder_lift", 
                  "state.main_elbow_flex", "state.main_wrist_flex", 
                  "state.main_wrist_roll", "state.main_gripper"]
    action_keys = ["action.main_shoulder_pan", "action.main_shoulder_lift", 
                   "action.main_elbow_flex", "action.main_wrist_flex", 
                   "action.main_wrist_roll", "action.main_gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    
    # ... rest similar to So100DataConfig
```

### Custom Embodiment Tag
If your robot is slightly different from SO-100, add a new embodiment tag:

```python
# In gr00t/data/embodiment_tags.py
PAPER_RETURN = "paper_return"
```

## Step 5: Training and Evaluation

### Training Script
```bash
# Basic training
PYTHONPATH=. torchrun scripts/gr00t_finetune.py \
    --dataset-path "./paper_return_dataset" \
    --data-config so100 \
    --embodiment_tag "so100" \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --wandb-project "paper_return_training"
```

### Inference
```python
# inference_example.py
from gr00t.model.policy import GR00T_N1

# Load trained model
model = GR00T_N1.from_pretrained("path/to/your/checkpoint")

# Run inference on new observations
# ... (see getting_started/1_gr00t_inference.ipynb for complete example)
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

1. **Validate** your dataset loads correctly with the test script
2. **Start with small-scale training** (few epochs) to verify the pipeline works
3. **Scale up** with full training once everything is working
4. **Evaluate** your trained model on held-out test data
5. **Deploy** using the policy deployment guide (getting_started/5_policy_deployment.md)

This guide should get you started with using your `paper_return` dataset in the GR00T Dreams pipeline. The key advantage is that since your dataset is already SO-100 and LeRobot compatible, minimal preprocessing should be required.