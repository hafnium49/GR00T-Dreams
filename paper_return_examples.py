#!/usr/bin/env python3
"""
Paper Return Dataset Examples
============================

This script provides practical examples for working with the Hafnium49/paper_return dataset
in the GR00T Dreams framework.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def validate_dataset_structure(dataset_path: str) -> bool:
    """Validate that the dataset follows LeRobot v2.0 structure."""
    dataset_root = Path(dataset_path)
    
    # Check required files
    required_files = [
        "meta/episodes.jsonl",
        "meta/info.json", 
        "meta/tasks.jsonl",
        "meta/modality.json"
    ]
    
    print("🔍 Validating dataset structure...")
    missing_files = []
    for file_path in required_files:
        full_path = dataset_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
            
    if missing_files:
        print(f"  ❌ Missing required files: {missing_files}")
        return False
    
    # Check directory structure
    data_dir = dataset_root / "data" / "chunk-000"
    video_dir = dataset_root / "videos" / "chunk-000"
    
    if data_dir.exists():
        parquet_files = len(list(data_dir.glob('*.parquet')))
        print(f"  ✅ Found {parquet_files} parquet files in data/")
    else:
        print(f"  ❌ data/chunk-000/ directory missing")
        return False
    
    if video_dir.exists():
        video_dirs = list(video_dir.iterdir())
        print(f"  ✅ Found video directories: {[d.name for d in video_dirs]}")
    else:
        print(f"  ❌ videos/chunk-000/ directory missing")
        return False
        
    return True


def load_and_inspect_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load the dataset with GR00T and inspect its properties."""
    try:
        from gr00t.data.dataset import LeRobotSingleDataset
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        
        print("\n📊 Loading dataset with GR00T...")
        
        # Get SO100 data configuration
        so100_config = DATA_CONFIG_MAP["so100"]
        
        # Load dataset
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=so100_config.modality_config(),
            embodiment_tag=EmbodimentTag.SO100,
            transforms=so100_config.transform(),
        )
        
        print(f"  ✅ Dataset loaded successfully!")
        print(f"  📈 Number of episodes: {len(dataset)}")
        
        # Inspect sample data
        sample = dataset[0]
        print(f"  🔑 Sample data keys: {list(sample.keys())}")
        
        # Check dimensions
        info = {
            'num_episodes': len(dataset),
            'sample_keys': list(sample.keys())
        }
        
        for key in ['observation.state', 'action']:
            if key in sample:
                shape = sample[key].shape
                print(f"  📏 {key} shape: {shape}")
                info[f'{key}_shape'] = shape
        
        # Check video keys
        video_keys = [k for k in sample.keys() if 'observation.images' in k]
        for key in video_keys:
            shape = sample[key].shape
            print(f"  🎥 {key} shape: {shape}")
            info[f'{key}_shape'] = shape
            
        return info
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("  💡 Make sure you're in the GR00T Dreams conda environment")
        return {}
    except Exception as e:
        print(f"  ❌ Loading failed: {e}")
        return {}


def create_custom_modality_config(dataset_path: str, output_path: Optional[str] = None):
    """Create a custom modality.json file if needed."""
    if output_path is None:
        output_path = str(Path(dataset_path) / "meta" / "modality.json")
    
    # SO-100 compatible modality configuration
    modality_config = {
        "state": {
            "main_shoulder_pan": {"start": 0, "end": 1},
            "main_shoulder_lift": {"start": 1, "end": 2},
            "main_elbow_flex": {"start": 2, "end": 3},
            "main_wrist_flex": {"start": 3, "end": 4},
            "main_wrist_roll": {"start": 4, "end": 5},
            "main_gripper": {"start": 5, "end": 6}
        },
        "action": {
            "main_shoulder_pan": {"start": 0, "end": 1, "absolute": False},
            "main_shoulder_lift": {"start": 1, "end": 2, "absolute": False},
            "main_elbow_flex": {"start": 2, "end": 3, "absolute": False},
            "main_wrist_flex": {"start": 3, "end": 4, "absolute": False},
            "main_wrist_roll": {"start": 4, "end": 5, "absolute": False},
            "main_gripper": {"start": 5, "end": 6, "absolute": False}
        },
        "video": {
            "webcam": {"original_key": "observation.images.webcam"}
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        }
    }
    
    print(f"\n📝 Creating modality config at: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(modality_config, f, indent=2)
    
    print("  ✅ Modality config created!")
    print("  💡 Edit this file if your robot configuration differs from SO-100")


def show_training_commands(dataset_path: str):
    """Display training commands for the dataset."""
    print("\n🚀 Training Commands:")
    print("=" * 50)
    
    print("\n1. Quick Training (Direct fine-tuning):")
    print(f"   PYTHONPATH=. torchrun scripts/gr00t_finetune.py \\")
    print(f"       --dataset-path \"{dataset_path}\" \\")
    print(f"       --data-config so100 \\")
    print(f"       --embodiment_tag \"so100\" \\")
    print(f"       --batch-size 4 \\")
    print(f"       --learning-rate 1e-4 \\")
    print(f"       --num-epochs 100")
    
    print("\n2. Using Preprocessing Script:")
    print(f"   bash IDM_dump/scripts/preprocess/paper_return.sh")
    
    print("\n3. Using Fine-tuning Script:")
    print(f"   bash IDM_dump/scripts/finetune/paper_return.sh")
    
    print("\n4. Advanced IDM Pipeline:")
    print(f"   # First, convert to IDM format")
    print(f"   python IDM_dump/raw_to_lerobot.py \\")
    print(f"       --input_dir \"{dataset_path}\" \\")
    print(f"       --output_dir \"./paper_return_processed\" \\")
    print(f"       --cosmos_predict2")
    print(f"   ")
    print(f"   # Then extract actions with IDM")
    print(f"   python IDM_dump/dump_idm_actions.py \\")
    print(f"       --checkpoint \"seonghyeonye/IDM_so100\" \\")
    print(f"       --dataset \"./paper_return_processed\" \\")
    print(f"       --output_dir \"./paper_return_idm\" \\")
    print(f"       --num_gpus 1")
    print(f"   ")
    print(f"   # Finally, fine-tune on processed data")
    print(f"   PYTHONPATH=. torchrun scripts/gr00t_finetune.py \\")
    print(f"       --dataset-path \"./paper_return_idm\" \\")
    print(f"       --data-config so100 \\")
    print(f"       --embodiment_tag \"so100\"")


def main():
    """Main function to run all examples."""
    if len(sys.argv) != 2:
        print("Usage: python paper_return_examples.py <dataset_path>")
        print("Example: python paper_return_examples.py ./paper_return_dataset")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    print("🤖 Paper Return Dataset Integration Examples")
    print("=" * 50)
    
    # Step 1: Validate structure
    if not validate_dataset_structure(dataset_path):
        print("\n❌ Dataset validation failed!")
        print("💡 Make sure your dataset follows LeRobot v2.0 format")
        
        create_modality = input("\nWould you like to create a default modality.json? (y/n): ")
        if create_modality.lower() == 'y':
            create_custom_modality_config(dataset_path)
        
        return
    
    # Step 2: Load and inspect
    dataset_info = load_and_inspect_dataset(dataset_path)
    
    if not dataset_info:
        print("\n❌ Could not load dataset with GR00T")
        return
    
    # Step 3: Show training commands
    show_training_commands(dataset_path)
    
    print("\n📚 Additional Resources:")
    print("  - Full integration guide: PAPER_RETURN_INTEGRATION_GUIDE.md")
    print("  - Getting started notebooks: getting_started/")
    print("  - GR00T inference example: getting_started/1_gr00t_inference.ipynb")
    print("  - Policy deployment: getting_started/5_policy_deployment.md")


if __name__ == "__main__":
    main()