#!/usr/bin/env python
"""
Simple test script to run Cosmos Predict2 Video2World inference
"""

import sys
import os
from pathlib import Path

# Add cosmos-predict2 to Python path
sys.path.insert(0, '/home/hafnium/cosmos-predict2')

# Set environment variables for better performance
os.environ['PYTHONPATH'] = '/home/hafnium/cosmos-predict2:' + os.environ.get('PYTHONPATH', '')

# Now import the cosmos modules
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
import torch

def main():
    """Run a simple test of Cosmos Predict2 Video2World"""

    # Setup paths
    dataset_dir = Path("paper_return_filtered_dataset")
    output_dir = Path("results/paper_dreams_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Cosmos Predict2 Video2World Test ===")
    print(f"Using first frame from: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    # Extract first frame from dataset
    import cv2
    video_files = list((dataset_dir / "videos").glob("**/*.mp4"))
    if not video_files:
        print("Error: No videos found in dataset!")
        return 1

    video_path = video_files[0]
    print(f"\n1. Using video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame from {video_path}")
        return 1

    # Save first frame
    first_frame_path = output_dir / "first_frame.jpg"
    cv2.imwrite(str(first_frame_path), frame)
    print(f"   Saved first frame to: {first_frame_path}")

    # Create prompt
    prompt = """A high-definition video of a robotic arm performing paper manipulation task.
    The robot carefully picks up a white paper and places it precisely into a red square
    target area on the table. The movements are smooth and controlled, showing the precision
    of the robotic system. The camera captures the entire workspace clearly."""

    print(f"\n2. Using prompt: {prompt[:100]}...")

    # Load the model
    print("\n3. Loading Cosmos Predict2 2B Video2World model...")

    try:
        # Create the video generation pipeline
        pipe = Video2WorldPipeline.from_config(
            config=get_cosmos_predict2_video2world_pipeline(model_size="2B"),
            dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B"),
        )
        print("   Model loaded successfully!")

        # Generate video
        print("\n4. Generating video...")
        print("   Parameters: 256x256, 16 frames, 50 inference steps")

        outputs = pipe.generate(
            prompt=prompt,
            image=str(first_frame_path),
            height=256,
            width=256,
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=torch.Generator().manual_seed(42),
        )

        # Save the generated video
        output_video_path = output_dir / "generated_video.mp4"
        save_image_or_video(outputs.videos[0], str(output_video_path))

        print(f"\n✅ Success! Generated video saved to: {output_video_path}")
        print("\nNext steps:")
        print("1. Review the generated video")
        print("2. Run batch generation for more variations")
        print("3. Process with IDM for action extraction")

    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())