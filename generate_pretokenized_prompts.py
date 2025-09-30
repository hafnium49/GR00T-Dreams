#!/usr/bin/env python
"""
Generate pre-tokenized T5 embeddings using HuggingFace API
This avoids downloading the 45GB T5-11B model locally
"""

import json
import numpy as np
import requests
from typing import List, Dict
import pickle
import os

# Option 1: Use HuggingFace Inference API (requires API token)
def tokenize_with_hf_api(prompts: List[str], hf_token: str = None):
    """
    Use HuggingFace's Inference API to tokenize prompts with T5-11B

    Args:
        prompts: List of text prompts
        hf_token: HuggingFace API token (get from https://huggingface.co/settings/tokens)

    Returns:
        Dict with tokenized embeddings
    """

    API_URL = "https://api-inference.huggingface.co/models/google/t5-11b"

    if not hf_token:
        print("⚠️ No HuggingFace token provided.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Set it as environment variable: export HF_TOKEN='your_token'")
        hf_token = os.environ.get('HF_TOKEN', '')

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    tokenized_prompts = {}

    for i, prompt in enumerate(prompts):
        print(f"Tokenizing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

        try:
            # Request feature extraction (embeddings)
            payload = {
                "inputs": prompt,
                "options": {"use_cache": True, "wait_for_model": True}
            }

            response = requests.post(API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                embeddings = response.json()
                tokenized_prompts[f"prompt_{i}"] = {
                    "text": prompt,
                    "embeddings": embeddings,
                    "source": "hf_api"
                }
                print(f"  ✅ Tokenized successfully")
            else:
                print(f"  ❌ API error: {response.status_code}")
                print(f"     {response.text}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    return tokenized_prompts


# Option 2: Use smaller Flan-T5 model locally (3GB instead of 45GB)
def tokenize_with_flan_t5_local(prompts: List[str]):
    """
    Use Flan-T5-XL (3GB) locally as a substitute for T5-11B
    Better quality than T5 and much smaller
    """
    try:
        from transformers import T5TokenizerFast, T5EncoderModel
        import torch

        print("Loading Flan-T5-XL model (3GB)...")
        tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-xl")
        model = T5EncoderModel.from_pretrained("google/flan-t5-xl")

        if torch.cuda.is_available():
            model = model.to('cuda')
            print("  Using GPU for encoding")

        tokenized_prompts = {}

        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                print(f"Encoding prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)

                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}

                # Get encoder outputs
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state

                # Move to CPU and convert to numpy
                embeddings_np = embeddings.cpu().numpy()

                tokenized_prompts[f"prompt_{i}"] = {
                    "text": prompt,
                    "embeddings": embeddings_np.tolist(),
                    "shape": list(embeddings_np.shape),
                    "source": "flan_t5_xl_local"
                }
                print(f"  ✅ Encoded: shape {embeddings_np.shape}")

        return tokenized_prompts

    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers torch")
        return {}


# Option 3: Create compatible mock embeddings for testing
def create_mock_embeddings(prompts: List[str]):
    """
    Create mock embeddings that match T5 output dimensions
    For testing without actual model
    """
    # T5-11B output dimensions
    SEQUENCE_LENGTH = 77
    HIDDEN_DIM = 1024  # T5-11B hidden dimension

    tokenized_prompts = {}

    for i, prompt in enumerate(prompts):
        # Generate deterministic embeddings based on prompt
        np.random.seed(hash(prompt) % 10000)

        # Create embeddings with correct shape
        embeddings = np.random.randn(1, SEQUENCE_LENGTH, HIDDEN_DIM).astype(np.float32)

        # Add some structure to make it more realistic
        # Gradually decay attention over sequence
        for j in range(SEQUENCE_LENGTH):
            embeddings[0, j, :] *= np.exp(-j * 0.01)

        tokenized_prompts[f"prompt_{i}"] = {
            "text": prompt,
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "source": "mock"
        }
        print(f"Created mock embedding for: {prompt[:50]}...")

    return tokenized_prompts


def save_tokenized_prompts(tokenized_prompts: Dict, output_path: str = "pretokenized_prompts.json"):
    """
    Save tokenized prompts to file
    """
    # Convert numpy arrays to lists for JSON serialization
    for key in tokenized_prompts:
        if isinstance(tokenized_prompts[key]["embeddings"], np.ndarray):
            tokenized_prompts[key]["embeddings"] = tokenized_prompts[key]["embeddings"].tolist()

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(tokenized_prompts, f, indent=2)

    print(f"\n✅ Saved {len(tokenized_prompts)} tokenized prompts to {output_path}")

    # Also save as pickle for faster loading
    pickle_path = output_path.replace('.json', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(tokenized_prompts, f)
    print(f"✅ Also saved as pickle: {pickle_path}")


def main():
    """
    Generate pre-tokenized prompts for paper manipulation task
    """

    # Define prompts for SO-101 paper manipulation
    prompts = [
        "A robotic arm picks up white paper and places it into a red square target area on the table.",
        "High-definition video of SO-101 robot manipulating paper with precise movements.",
        "Robot gripper carefully grasps white paper sheet and moves it to designated red square zone.",
        "Automated paper handling task: robot arm transfers white sheet to red target area with smooth motion.",
        "Close-up view of robotic gripper picking up paper and placing it accurately in red square.",
        "SO-101 robot demonstrates paper manipulation skill by moving sheet into target zone.",
        "Time-lapse of robot repeatedly picking and placing paper in red square area.",
        "Multiple camera angles showing robot arm paper handling and placement task.",
        "Slow motion capture of gripper grasping paper edges and repositioning to target.",
        "Robot learning demonstration: paper pickup and precise placement in marked area.",
    ]

    print("=" * 60)
    print("Pre-tokenized Prompt Generator for Cosmos Predict2")
    print("=" * 60)

    # Choose method
    print("\nSelect tokenization method:")
    print("1. HuggingFace API (requires token, no download)")
    print("2. Flan-T5-XL local (3GB download)")
    print("3. Mock embeddings (for testing)")

    choice = input("\nEnter choice (1-3) [default: 3]: ").strip() or "3"

    if choice == "1":
        # Use HuggingFace API
        hf_token = input("Enter HuggingFace token (or press Enter to use HF_TOKEN env var): ").strip()
        tokenized_prompts = tokenize_with_hf_api(prompts, hf_token)

    elif choice == "2":
        # Use local Flan-T5
        tokenized_prompts = tokenize_with_flan_t5_local(prompts)

    else:
        # Use mock embeddings
        print("\nGenerating mock embeddings for testing...")
        tokenized_prompts = create_mock_embeddings(prompts)

    # Save results
    if tokenized_prompts:
        save_tokenized_prompts(tokenized_prompts)

        # Print summary
        print("\n" + "=" * 60)
        print("Summary of tokenized prompts:")
        print("=" * 60)
        for key, data in tokenized_prompts.items():
            print(f"\n{key}:")
            print(f"  Text: {data['text'][:60]}...")
            if 'shape' in data:
                print(f"  Shape: {data['shape']}")
            print(f"  Source: {data['source']}")
    else:
        print("\n❌ No prompts were tokenized")


if __name__ == "__main__":
    main()