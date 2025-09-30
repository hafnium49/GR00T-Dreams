#!/usr/bin/env python
"""
Simple script to generate T5 embeddings using HuggingFace Inference API
No local model download required!
"""

import requests
import json
import numpy as np
import os

# HuggingFace API configuration
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/google/t5-11b"
# Alternative: Use Flan-T5 which is often better
# HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/google/flan-t5-xl"

def get_t5_embeddings_from_api(text, hf_token=None):
    """
    Get T5 embeddings from HuggingFace Inference API

    Args:
        text: Input text to encode
        hf_token: HuggingFace API token (optional for public models)

    Returns:
        numpy array of embeddings
    """

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        embeddings = response.json()
        return np.array(embeddings)
    else:
        print(f"API Error {response.status_code}: {response.text}")
        return None


def generate_cosmos_prompts_embeddings():
    """
    Generate embeddings for Cosmos Predict2 prompts for paper manipulation
    """

    # Get HF token from environment or prompt
    hf_token = os.environ.get('HF_TOKEN', '')
    if not hf_token:
        print("No HF_TOKEN environment variable found.")
        print("You can get a token from: https://huggingface.co/settings/tokens")
        print("Or continue without token (may have rate limits)")
        use_token = input("Enter token (or press Enter to continue without): ").strip()
        if use_token:
            hf_token = use_token

    # Prompts for paper manipulation task
    prompts = {
        "pickup_paper": "A robotic arm picks up white paper and places it into a red square target area",
        "precise_movement": "SO-101 robot manipulates paper with precise controlled movements",
        "gripper_action": "Robot gripper grasps paper and moves it to designated red square",
        "smooth_transfer": "Smooth paper transfer from pickup point to red target zone",
        "close_up_grasp": "Close-up of gripper carefully grasping paper edges",
        "multi_angle": "Multiple camera angles of paper manipulation task",
        "slow_motion": "Slow motion capture of paper pickup and placement",
        "repeat_task": "Robot repeatedly picks and places paper in target area",
        "learning_demo": "Robot learning demonstration of paper handling skill",
        "final_placement": "Final precise placement of paper in center of red square"
    }

    print("Generating T5 embeddings for Cosmos prompts...")
    print("=" * 60)

    embeddings_dict = {}

    for key, prompt in prompts.items():
        print(f"\nProcessing: {key}")
        print(f"  Prompt: {prompt[:60]}...")

        embeddings = get_t5_embeddings_from_api(prompt, hf_token)

        if embeddings is not None:
            embeddings_dict[key] = {
                "prompt": prompt,
                "embeddings": embeddings.tolist(),
                "shape": list(embeddings.shape)
            }
            print(f"  ✅ Success! Shape: {embeddings.shape}")
        else:
            print(f"  ❌ Failed to get embeddings")

    # Save to file
    if embeddings_dict:
        output_file = "cosmos_prompt_embeddings.json"
        with open(output_file, 'w') as f:
            json.dump(embeddings_dict, f, indent=2)
        print(f"\n✅ Saved {len(embeddings_dict)} embeddings to {output_file}")

        # Also save as numpy format for faster loading
        np_file = "cosmos_prompt_embeddings.npz"
        np_arrays = {k: np.array(v['embeddings']) for k, v in embeddings_dict.items()}
        np.savez_compressed(np_file, **np_arrays)
        print(f"✅ Also saved as compressed numpy: {np_file}")

        return embeddings_dict
    else:
        print("\n❌ No embeddings were generated")
        return None


def load_and_use_embeddings():
    """
    Example of loading and using the pre-tokenized embeddings
    """

    # Load from JSON
    with open("cosmos_prompt_embeddings.json", 'r') as f:
        embeddings_dict = json.load(f)

    print("\nLoaded embeddings:")
    for key, data in embeddings_dict.items():
        embeddings = np.array(data['embeddings'])
        print(f"  {key}: shape {embeddings.shape}")

    # Or load from numpy (faster)
    npz_data = np.load("cosmos_prompt_embeddings.npz")
    print("\nNumpy arrays available:", list(npz_data.keys()))

    return embeddings_dict


if __name__ == "__main__":
    print("=" * 60)
    print("HuggingFace API T5 Tokenizer for Cosmos Predict2")
    print("=" * 60)

    # Generate embeddings
    embeddings = generate_cosmos_prompts_embeddings()

    if embeddings:
        print("\n" + "=" * 60)
        print("Embeddings generated successfully!")
        print("Use these in your Cosmos inference notebook")
        print("=" * 60)

        # Example usage
        print("\nExample usage in notebook:")
        print("```python")
        print("# Load pre-tokenized embeddings")
        print("import json, numpy as np")
        print("with open('cosmos_prompt_embeddings.json', 'r') as f:")
        print("    embeddings = json.load(f)")
        print("")
        print("# Use in pipeline")
        print("prompt_embeddings = np.array(embeddings['pickup_paper']['embeddings'])")
        print("# Pass to Cosmos pipeline instead of text prompt")
        print("```")