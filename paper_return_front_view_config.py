#!/usr/bin/env python3
"""
Paper Return Front View Dataset Configuration
===========================================

Configuration for handling the dual-camera paper_return_front_view dataset,
focusing only on the main camera and filtering out secondary_0 videos.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def filter_main_camera_only(dataset_path: str, output_path: Optional[str] = None):
    """
    Filter the dataset to use only main camera videos, excluding secondary_0.
    
    Args:
        dataset_path: Path to the paper_return_front_view dataset
        output_path: Optional output path for filtered dataset
    """
    dataset_root = Path(dataset_path)
    
    if output_path is None:
        output_path = str(dataset_root.parent / f"{dataset_root.name}_main_only")
    
    output_root = Path(output_path)
    
    print(f"🎥 Filtering dataset to main camera only...")
    print(f"   Source: {dataset_path}")
    print(f"   Output: {output_path}")
    
    # Copy directory structure
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Copy meta files
    meta_src = dataset_root / "meta"
    meta_dst = output_root / "meta"
    if meta_src.exists():
        shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)
        print("   ✅ Copied metadata files")
    
    # Copy data files
    data_src = dataset_root / "data"
    data_dst = output_root / "data"
    if data_src.exists():
        shutil.copytree(data_src, data_dst, dirs_exist_ok=True)
        print("   ✅ Copied data files")
    
    # Filter video directories (exclude secondary_0)
    videos_src = dataset_root / "videos"
    videos_dst = output_root / "videos"
    
    if videos_src.exists():
        videos_dst.mkdir(parents=True, exist_ok=True)
        
        for chunk_dir in videos_src.iterdir():
            if chunk_dir.is_dir():
                chunk_dst = videos_dst / chunk_dir.name
                chunk_dst.mkdir(parents=True, exist_ok=True)
                
                # Filter out secondary_0 directories
                for episode_dir in chunk_dir.iterdir():
                    if episode_dir.is_dir() and "secondary_0" not in episode_dir.name:
                        episode_dst = chunk_dst / episode_dir.name
                        shutil.copytree(episode_dir, episode_dst, dirs_exist_ok=True)
                        print(f"   ✅ Copied {episode_dir.name} (main camera)")
                    elif "secondary_0" in episode_dir.name:
                        print(f"   ⏭️  Skipped {episode_dir.name} (secondary camera)")
        
        print("   ✅ Filtered video directories")
    
    # Update modality.json for single camera
    update_modality_for_single_camera(output_root)
    
    print(f"\n✅ Dataset filtered successfully!")
    print(f"   Main camera dataset ready at: {output_path}")
    return output_path


def update_modality_for_single_camera(dataset_root: Path):
    """Update modality.json to reflect single camera setup."""
    modality_file = dataset_root / "meta" / "modality.json"
    
    if not modality_file.exists():
        print("   ⚠️  No modality.json found, creating default...")
        create_single_camera_modality(modality_file)
        return
    
    try:
        with open(modality_file, 'r') as f:
            modality = json.load(f)
        
        # Ensure video section uses main camera only
        if "video" in modality:
            # Remove any secondary camera references
            video_keys = list(modality["video"].keys())
            for key in video_keys:
                if "secondary" in key.lower():
                    del modality["video"][key]
                    print(f"   🗑️  Removed secondary camera key: {key}")
            
            # Ensure main camera key exists
            if "webcam" not in modality["video"]:
                modality["video"]["webcam"] = {"original_key": "observation.images.webcam"}
                print("   ➕ Added main webcam configuration")
        
        # Write updated modality
        with open(modality_file, 'w') as f:
            json.dump(modality, f, indent=2)
        
        print("   ✅ Updated modality.json for single camera")
        
    except Exception as e:
        print(f"   ⚠️  Error updating modality.json: {e}")
        print("   💡 Creating new single camera modality...")
        create_single_camera_modality(modality_file)


def create_single_camera_modality(modality_file: Path):
    """Create a single camera modality.json file."""
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
    
    modality_file.parent.mkdir(parents=True, exist_ok=True)
    with open(modality_file, 'w') as f:
        json.dump(modality_config, f, indent=2)
    
    print("   ✅ Created single camera modality.json")


def main():
    """Main function for dataset filtering."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python paper_return_front_view_config.py <dataset_path> [output_path]")
        print("Example: python paper_return_front_view_config.py ./paper_return_front_view_dataset")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    filtered_path = filter_main_camera_only(dataset_path, output_path)
    
    print(f"\n🎉 Ready to use filtered dataset:")
    print(f"   python paper_return_examples.py {filtered_path}")


if __name__ == "__main__":
    main()