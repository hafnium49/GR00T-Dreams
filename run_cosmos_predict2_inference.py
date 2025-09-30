#!/usr/bin/env python3
"""
Script to run Cosmos Predict2 inference on paper_return_filtered_dataset
This generates synthetic video variations for data augmentation
"""

import os
import sys
import torch
from pathlib import Path
import json
from PIL import Image
import numpy as np

# Add cosmos-predict2 to path
sys.path.append('/home/hafnium/cosmos-predict2')

from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

def extract_first_frame(video_dir):
    """Extract the first frame from a video to use as input"""
    import cv2

    # Find first video file
    video_files = list(Path(video_dir).glob("**/*.mp4"))
    if not video_files:
        raise ValueError(f"No MP4 files found in {video_dir}")

    video_path = video_files[0]
    print(f"Using video: {video_path}")

    # Extract first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame from {video_path}")

    # Convert BGR to RGB and save
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    first_frame_path = "temp_first_frame.jpg"
    Image.fromarray(frame).save(first_frame_path)

    return first_frame_path

def generate_task_prompts(dataset_path):
    """Generate task-specific prompts based on dataset metadata"""
    prompts = []

    # Check if we have task descriptions
    tasks_file = Path(dataset_path) / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file, 'r') as f:
            for line in f:
                task = json.loads(line)
                task_name = task.get('task', 'paper manipulation')
                # Generate variations of the task
                base_prompt = f"A robotic arm performing {task_name} task. "
                prompts.append(base_prompt + "The robot moves white paper into a red square target area on a table surface. High-quality video showing precise manipulation.")
    else:
        # Default prompts for paper return task
        prompts = [
            "A robotic arm performing paper manipulation task. The robot moves white paper into a red square target area on a table surface with precise movements.",
            "High-definition video of a robotic arm carefully grasping and placing white paper into a designated red square. The movements are smooth and controlled.",
            "Industrial robotic arm demonstration: picking up white paper and accurately placing it in the center of a red square target area.",
        ]

    return prompts

def run_cosmos_inference(dataset_path, output_dir, model_size="2B", num_variations=3):
    """
    Run Cosmos Predict2 Video2World inference on paper_return dataset

    Args:
        dataset_path: Path to paper_return_filtered_dataset
        output_dir: Directory to save generated videos
        model_size: Model size to use (2B or 14B)
        num_variations: Number of variations to generate per prompt
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Cosmos Predict2 {model_size} Video2World model...")

    # Create the video generation pipeline
    pipe = Video2WorldPipeline.from_config(
        config=get_cosmos_predict2_video2world_pipeline(model_size=model_size),
        dit_path=get_cosmos_predict2_video2world_checkpoint(model_size=model_size),
    )

    # Extract first frame from dataset videos
    video_dir = Path(dataset_path) / "videos"
    print(f"Extracting first frame from videos in {video_dir}...")
    first_frame_path = extract_first_frame(video_dir)

    # Generate task-specific prompts
    prompts = generate_task_prompts(dataset_path)

    print(f"Generating {num_variations} variations for {len(prompts)} prompts...")

    # Generate videos for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}: {prompt[:100]}...")

        for variation_idx in range(num_variations):
            print(f"  Generating variation {variation_idx + 1}/{num_variations}...")

            # Generate video
            outputs = pipe.generate(
                prompt=prompt,
                image=first_frame_path,
                height=256,  # Adjust based on your needs
                width=256,   # Adjust based on your needs
                num_frames=16,  # Number of frames to generate
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=torch.Generator().manual_seed(42 + variation_idx),
            )

            # Save the generated video
            output_filename = output_path / f"dream_p{prompt_idx:02d}_v{variation_idx:02d}.mp4"
            save_image_or_video(outputs.videos[0], str(output_filename))
            print(f"    Saved: {output_filename}")

    # Clean up temporary file
    if Path(first_frame_path).exists():
        Path(first_frame_path).unlink()

    print(f"\n✅ Generated {len(prompts) * num_variations} synthetic videos in {output_dir}")
    print(f"Next steps:")
    print(f"1. Convert these videos for IDM processing")
    print(f"2. Extract actions using IDM")
    print(f"3. Merge with original dataset for training")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Cosmos Predict2 inference on paper_return dataset")
    parser.add_argument("--dataset-path", type=str, default="paper_return_filtered_dataset",
                        help="Path to paper_return_filtered_dataset")
    parser.add_argument("--output-dir", type=str, default="results/paper_dreams",
                        help="Output directory for generated videos")
    parser.add_argument("--model-size", type=str, default="2B", choices=["2B", "14B"],
                        help="Model size to use")
    parser.add_argument("--num-variations", type=int, default=3,
                        help="Number of variations to generate per prompt")

    args = parser.parse_args()

    # Run inference
    run_cosmos_inference(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        num_variations=args.num_variations
    )