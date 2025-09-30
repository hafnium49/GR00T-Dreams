#!/usr/bin/env python
"""
Cosmos Predict2 inference with smaller T5 model for practical use
"""

import os
import sys
sys.path.insert(0, '/home/hafnium/cosmos-predict2')

# Monkey-patch the T5 model path to use a smaller model
import cosmos_predict2.configs.base.config_video2world as config_module

# Override the T5 checkpoint path
original_get_config = config_module.get_cosmos_predict2_video2world_pipeline

def get_patched_config(model_size="2B"):
    config = original_get_config(model_size)
    # Use Flan-T5-XL instead of T5-11B (3GB vs 45GB)
    config.text_encoder.t5.ckpt_path = "google/flan-t5-xl"
    return config

config_module.get_cosmos_predict2_video2world_pipeline = get_patched_config

# Now run the inference
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
import torch

def main():
    print("=== Cosmos Predict2 with Flan-T5-XL ===")
    print("Using smaller, more efficient T5 model")

    # Check CUDA
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Input/output paths
    input_image = "/home/hafnium/GR00T-Dreams/test_input_frame.jpg"
    output_video = "cosmos_output_small.mp4"

    # Extract frame if needed
    if not os.path.exists(input_image):
        import cv2
        from pathlib import Path
        video_files = list(Path("paper_return_filtered_dataset/videos").glob("**/*.mp4"))
        if video_files:
            cap = cv2.VideoCapture(str(video_files[0]))
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(input_image, frame)
                print(f"Extracted frame: {input_image}")
            cap.release()

    # Create pipeline with patched config
    print("\nLoading model with Flan-T5-XL...")
    pipe = Video2WorldPipeline.from_config(
        config=get_patched_config(model_size="2B"),
        dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B"),
    )

    print("Model loaded!")

    # Generate
    print("\nGenerating video...")
    prompt = "A robotic arm picks up white paper and places it in a red square target area"

    outputs = pipe.generate(
        prompt=prompt,
        image=input_image,
        height=256,
        width=256,
        num_frames=4,  # Start small
        guidance_scale=7.5,
        num_inference_steps=10,  # Quick test
    )

    # Save
    save_image_or_video(outputs.videos[0], output_video)
    print(f"✅ Saved to: {output_video}")

if __name__ == "__main__":
    main()