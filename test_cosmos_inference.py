#!/usr/bin/env python
"""
Test Cosmos Predict2 Video2World inference on paper_return_filtered_dataset
This script should be run with: conda run -n cosmos python test_cosmos_inference.py
"""

import sys
import os
from pathlib import Path

# Add cosmos-predict2 to Python path
sys.path.insert(0, '/home/hafnium/cosmos-predict2')

def test_inference():
    print("=== Cosmos Predict2 Video2World Inference Test ===")

    # Step 1: Extract first frame from dataset
    print("\n1. Extracting first frame from dataset...")
    import cv2

    dataset_dir = Path("paper_return_filtered_dataset")
    video_files = list((dataset_dir / "videos").glob("**/*.mp4"))

    if not video_files:
        print("Error: No videos found in dataset!")
        return 1

    video_path = video_files[0]
    print(f"   Using video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame from {video_path}")
        return 1

    # Save first frame
    first_frame_path = "test_input_frame.jpg"
    cv2.imwrite(first_frame_path, frame)
    print(f"   Saved first frame to: {first_frame_path}")

    # Step 2: Setup and run inference
    print("\n2. Loading Cosmos Predict2 model...")

    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")

        # Import Cosmos modules
        from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
        from imaginaire.utils.io import save_image_or_video
        from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

        # Create the video generation pipeline
        print("\n3. Creating Video2World pipeline (2B model)...")
        pipe = Video2WorldPipeline.from_config(
            config=get_cosmos_predict2_video2world_pipeline(model_size="2B"),
            dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B"),
        )
        print("   Pipeline created successfully!")

        # Generate video
        print("\n4. Generating synthetic video...")
        prompt = """A high-definition video of a robotic arm performing paper manipulation task.
        The SO-101 robot carefully picks up a white paper and places it precisely into a red square
        target area on the table. The movements are smooth and controlled."""

        print(f"   Prompt: {prompt[:80]}...")
        print("   Parameters: 256x256, 8 frames, 25 inference steps")

        outputs = pipe.generate(
            prompt=prompt,
            image=first_frame_path,
            height=256,
            width=256,
            num_frames=8,  # Start with fewer frames for testing
            guidance_scale=7.5,
            num_inference_steps=25,  # Fewer steps for faster testing
            generator=torch.Generator().manual_seed(42),
        )

        # Save the generated video
        output_video_path = "test_generated_video.mp4"
        save_image_or_video(outputs.videos[0], output_video_path)

        print(f"\n✅ Success! Generated video saved to: {output_video_path}")
        print("\n=== Test Summary ===")
        print(f"Input frame: {first_frame_path}")
        print(f"Output video: {output_video_path}")
        print(f"Model: Cosmos Predict2 2B Video2World")
        print(f"Resolution: 256x256, Frames: 8")

        print("\n=== Next Steps ===")
        print("1. Review the generated video")
        print("2. If successful, run batch generation with more frames:")
        print("   python run_cosmos_predict2_inference.py")
        print("3. Process generated videos through IDM for action extraction")
        print("4. Merge with original dataset for SO-101 training")

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  conda run -n cosmos pip install loguru hydra-core omegaconf")
        return 1
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(test_inference())