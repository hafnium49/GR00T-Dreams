#!/usr/bin/env python
"""
Fixed Cosmos Predict2 Video2World example for paper_return dataset
Run with: conda activate cosmos && python run_cosmos_video2world.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add local cosmos-predict2 to path (prioritize local over installed)
sys.path.insert(0, '/home/hafnium/cosmos-predict2')

# Now import the modules
from imaginaire.constants import (
    get_cosmos_predict2_video2world_checkpoint,
    # Remove print_environment_info if it causes issues
)
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

def main(args):
    """Run Cosmos Predict2 Video2World inference."""

    print("=" * 50)
    print("Cosmos Predict2 Video2World Inference")
    print("=" * 50)

    # Print environment info if available
    try:
        from imaginaire.constants import print_environment_info
        print_environment_info()
    except:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    # Check if input image exists
    if not Path(args.image).exists():
        print(f"\n⚠️ Input image not found: {args.image}")
        print("Please provide a valid input image path")

        # Try to extract from dataset
        import cv2
        video_files = list(Path("paper_return_filtered_dataset/videos").glob("**/*.mp4"))
        if video_files:
            video_path = video_files[0]
            print(f"Extracting frame from: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            if ret:
                args.image = "extracted_frame.jpg"
                cv2.imwrite(args.image, frame)
                print(f"Saved frame to: {args.image}")
            cap.release()
        else:
            return 1

    # Read prompt
    if Path(args.prompt).exists():
        with open(args.prompt, 'r') as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    print(f"\nPrompt: {prompt[:100]}...")
    print(f"Input image: {args.image}")
    print(f"Output path: {args.save_path}")
    print(f"Model size: {args.model_size}")

    # Create the video generation pipeline
    print("\nLoading model...")
    pipe = Video2WorldPipeline.from_config(
        config=get_cosmos_predict2_video2world_pipeline(model_size=args.model_size),
        dit_path=get_cosmos_predict2_video2world_checkpoint(model_size=args.model_size),
    )

    print("Model loaded successfully!")

    # Generate video
    print("\nGenerating video...")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")

    outputs = pipe.generate(
        prompt=prompt,
        image=args.image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )

    # Save the generated video
    save_image_or_video(outputs.videos[0], args.save_path)
    print(f"\n✅ Video saved to: {args.save_path}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmos Predict2 Video2World Generation")

    parser.add_argument(
        "--model_size",
        type=str,
        default="2B",
        choices=["2B", "14B"],
        help="Model size to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A robotic arm performing paper manipulation task. The robot picks up white paper and places it into a red square target area on the table.",
        help="Text prompt or path to prompt file"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_input_frame.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output_video.mp4",
        help="Path to save output video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Video width"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )

    args = parser.parse_args()
    sys.exit(main(args))