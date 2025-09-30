#!/usr/bin/env python
"""
Working Cosmos Predict2 inference script for paper_return dataset
Run with: conda activate cosmos && python cosmos_inference_working.py
"""

import os
import sys
import torch
from pathlib import Path

# Add cosmos-predict2 to path
sys.path.insert(0, '/home/hafnium/cosmos-predict2')

def main():
    print("=== Cosmos Predict2 Video2World Inference ===\n")

    # Check environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Import the necessary modules directly
    try:
        from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
        from imaginaire.utils.io import save_image_or_video
        from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("\nTrying alternative imports...")

        # Try to import with minimal dependencies
        import imaginaire
        import cosmos_predict2

        # Use the modules directly
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
        from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline

    print("\n✅ Modules imported successfully!")

    # Setup paths
    input_image = "/home/hafnium/GR00T-Dreams/test_input_frame.jpg"
    output_video = "cosmos_output.mp4"

    # Check if input exists
    if not Path(input_image).exists():
        print(f"\n⚠️ Input image not found: {input_image}")
        print("Extracting first frame from dataset...")

        import cv2
        video_files = list(Path("paper_return_filtered_dataset/videos").glob("**/*.mp4"))
        if video_files:
            video_path = video_files[0]
            print(f"Using video: {video_path}")
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(input_image, frame)
                print(f"Saved frame to: {input_image}")
            cap.release()

    print(f"\nInput image: {input_image}")
    print(f"Output will be saved to: {output_video}")

    # Create the pipeline
    print("\n🔧 Creating Video2World pipeline...")

    try:
        # Get model checkpoint path
        model_size = "2B"

        # Try to get the checkpoint path
        try:
            dit_path = get_cosmos_predict2_video2world_checkpoint(model_size=model_size)
        except:
            # Fallback to HuggingFace model
            dit_path = f"nvidia/Cosmos-Predict2-{model_size}-Video2World"
            print(f"Using HuggingFace model: {dit_path}")

        # Create pipeline config
        config = get_cosmos_predict2_video2world_pipeline(model_size=model_size)

        # Create the pipeline
        pipe = Video2WorldPipeline.from_config(
            config=config,
            dit_path=dit_path,
        )

        print("✅ Pipeline created successfully!")

        # Generate video
        print("\n🎬 Generating video...")
        print("  Prompt: Robot arm manipulating paper into red square")
        print("  Resolution: 256x256")
        print("  Frames: 8")
        print("  This may take a few minutes...")

        prompt = "A robotic arm performing paper manipulation task. The robot picks up white paper and places it into a red square target area."

        outputs = pipe.generate(
            prompt=prompt,
            image=input_image,
            height=256,
            width=256,
            num_frames=8,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator().manual_seed(42),
        )

        # Save the output
        if hasattr(outputs, 'videos'):
            video = outputs.videos[0]
        else:
            video = outputs[0]

        # Save using imageio or direct tensor save
        try:
            from imaginaire.utils.io import save_image_or_video
            save_image_or_video(video, output_video)
        except:
            import imageio
            import numpy as np

            # Convert tensor to numpy if needed
            if torch.is_tensor(video):
                video = video.cpu().numpy()

            # Ensure proper shape and type
            if video.dtype != np.uint8:
                video = (video * 255).astype(np.uint8)

            # Save with imageio
            imageio.mimsave(output_video, video, fps=8)

        print(f"\n✅ Success! Video saved to: {output_video}")

    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you're in the cosmos conda environment")
        print("2. Try installing missing dependencies:")
        print("   conda run -n cosmos pip install imageio transformers diffusers")
        print("3. Check CUDA compatibility:")
        print("   nvidia-smi")

if __name__ == "__main__":
    main()