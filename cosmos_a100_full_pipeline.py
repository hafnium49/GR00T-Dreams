#!/usr/bin/env python3
"""
Full Cosmos Predict2 pipeline for A100 GPU (Colab/Jupyter compatible).
Handles both T5 encoding and video generation in a single session.
Optimized for A100 with 40GB memory.
"""

import os
import gc
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class OptimizedT5Encoder:
    """Memory-optimized T5 encoder for A100."""

    def __init__(self, model_name: str = "google/flan-t5-xl", device: str = "cuda"):
        """
        Initialize T5 encoder with memory optimizations.

        Args:
            model_name: T5 model to use. Options:
                - "google/flan-t5-xl" (3GB, faster)
                - "google-t5/t5-11b" (22GB in FP16, better quality)
            device: Device to use
        """
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self, use_fp16: bool = True, use_8bit: bool = False):
        """Load T5 model with memory optimizations."""
        from transformers import T5TokenizerFast, T5EncoderModel

        print(f"Loading T5 tokenizer from {self.model_name}...")
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)

        print(f"Loading T5 encoder model (FP16={use_fp16}, 8bit={use_8bit})...")

        if use_8bit:
            # Use bitsandbytes for 8-bit quantization (requires bitsandbytes package)
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                self.model = T5EncoderModel.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            except ImportError:
                print("8-bit quantization not available, falling back to FP16")
                use_8bit = False

        if not use_8bit:
            self.model = T5EncoderModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if use_fp16 else torch.float32
            )
            self.model = self.model.to(self.device)

        self.model.eval()

        # Print memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def encode(self, prompts: Union[str, List[str]], max_length: int = 77) -> Dict:
        """Encode text prompts to embeddings."""
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize
        tokens = self.tokenizer(
            prompts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                encoder_outputs = self.model(**tokens)

        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "encoder_hidden_states": encoder_outputs.last_hidden_state
        }

    def unload(self):
        """Free memory by unloading the model."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("T5 encoder unloaded from memory")


class A100CosmosPipeline:
    """Cosmos Predict2 pipeline optimized for A100."""

    def __init__(self, model_size: str = "2B", checkpoint_dir: Optional[str] = None):
        """
        Initialize Cosmos pipeline.

        Args:
            model_size: Model size ("2B", "5B", "14B")
            checkpoint_dir: Path to checkpoint directory
        """
        self.model_size = model_size
        self.checkpoint_dir = checkpoint_dir or "/home/hafnium/cosmos-predict2/checkpoints"
        self.pipe = None
        self.t5_encoder = None

    def setup_t5_encoder(self, model_name: str = "google/flan-t5-xl"):
        """Setup T5 encoder for text encoding."""
        self.t5_encoder = OptimizedT5Encoder(model_name=model_name)
        self.t5_encoder.load(use_fp16=True)

    def setup_cosmos(self):
        """Setup Cosmos Predict2 pipeline."""
        try:
            from cosmos_predict2.inference import (
                Video2WorldPipeline,
                get_cosmos_predict2_video2world_pipeline,
                get_cosmos_predict2_video2world_checkpoint
            )

            print(f"Loading Cosmos Predict2 {self.model_size} model...")

            # Custom checkpoint path if provided
            if os.path.exists(os.path.join(self.checkpoint_dir, "nvidia", f"Cosmos-Predict2-{self.model_size}-Video2World")):
                dit_path = os.path.join(
                    self.checkpoint_dir,
                    "nvidia",
                    f"Cosmos-Predict2-{self.model_size}-Video2World",
                    "model-720p-16fps.pt"
                )
            else:
                dit_path = get_cosmos_predict2_video2world_checkpoint(model_size=self.model_size)

            self.pipe = Video2WorldPipeline.from_config(
                config=get_cosmos_predict2_video2world_pipeline(model_size=self.model_size),
                dit_path=dit_path,
            )

            # Move to GPU
            self.pipe = self.pipe.to("cuda")
            self.pipe.eval()

            print(f"Cosmos pipeline loaded successfully")
            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        except Exception as e:
            print(f"Error loading Cosmos pipeline: {e}")
            raise

    def generate_video(
        self,
        input_video_path: str,
        prompt: str,
        num_frames: int = 121,
        fps: int = 16,
        use_cached_embeddings: bool = False,
        cached_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Generate video with optimized memory management.

        Args:
            input_video_path: Path to input video
            prompt: Text prompt for generation
            num_frames: Number of frames to generate
            fps: Frames per second
            use_cached_embeddings: Whether to use pre-computed embeddings
            cached_embeddings: Pre-computed T5 embeddings
        """
        import decord
        from einops import rearrange

        # Load input video
        print(f"Loading input video from {input_video_path}...")
        vr = decord.VideoReader(input_video_path)
        frames = vr[:].asnumpy()  # Get first frame or frames

        # Prepare input
        if len(frames.shape) == 3:  # Single frame
            frames = frames[np.newaxis, ...]  # Add batch dimension

        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = rearrange(frames_tensor, "t h w c -> 1 c t h w")
        frames_tensor = frames_tensor.to("cuda")

        # Handle text encoding
        if use_cached_embeddings and cached_embeddings is not None:
            print("Using cached T5 embeddings...")
            text_embeddings = cached_embeddings
        else:
            print(f"Encoding prompt: {prompt[:50]}...")
            if self.t5_encoder is None:
                self.setup_t5_encoder()

            encoded = self.t5_encoder.encode(prompt)
            text_embeddings = encoded["encoder_hidden_states"]

            # Optionally unload T5 to free memory for generation
            if self.model_size in ["5B", "14B"]:
                print("Unloading T5 to free memory for larger Cosmos model...")
                self.t5_encoder.unload()

        # Generate video
        print(f"Generating {num_frames} frames at {fps} FPS...")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = self.pipe(
                    frames_tensor,
                    text_embeddings,
                    num_frames=num_frames,
                    fps=fps,
                    seed=42
                )

        print("Video generation complete!")
        return output

    def cleanup(self):
        """Clean up resources."""
        if self.t5_encoder is not None:
            self.t5_encoder.unload()
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()


def check_a100():
    """Check if running on A100 and optimize settings."""
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! This script requires a GPU.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")

    if "A100" in gpu_name:
        print("✓ Running on A100 GPU - optimal configuration!")

        # Enable TF32 for better performance on A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Print memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU memory: {total_memory:.1f} GB")

        return True
    else:
        print(f"Note: Not running on A100. Performance may vary.")
        return False


def main():
    """Main function demonstrating full pipeline on A100."""

    # Check GPU
    is_a100 = check_a100()

    # Initialize pipeline
    pipeline = A100CosmosPipeline(model_size="2B")

    try:
        # Option 1: Use smaller Flan-T5-XL (3GB) - leaves more room for Cosmos
        print("\n" + "="*50)
        print("Option 1: Using Flan-T5-XL (3GB) for encoding")
        print("="*50)
        pipeline.setup_t5_encoder(model_name="google/flan-t5-xl")

        # Encode some prompts
        prompts = [
            "The robot picks up a piece of paper from the table",
            "The robot folds the paper in half"
        ]

        embeddings_cache = {}
        for prompt in prompts:
            encoded = pipeline.t5_encoder.encode(prompt)
            embeddings_cache[prompt] = encoded["encoder_hidden_states"]
            print(f"Encoded: {prompt[:30]}... Shape: {encoded['encoder_hidden_states'].shape}")

        # Setup Cosmos
        pipeline.setup_cosmos()

        print("\nMemory usage after loading both models:")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Option 2: For T5-11B on A100 (if you want better quality)
        if is_a100:
            print("\n" + "="*50)
            print("Option 2: A100 can handle T5-11B (22GB) + Cosmos-2B")
            print("="*50)
            print("To use T5-11B, change model_name to 'google-t5/t5-11b'")
            print("This will use ~30GB total, fitting within A100's 40GB")

    finally:
        # Cleanup
        pipeline.cleanup()
        print("\nPipeline cleaned up successfully")


if __name__ == "__main__":
    main()