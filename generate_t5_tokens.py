#!/usr/bin/env python
"""
Generate pre-tokenized T5 inputs that Cosmos Predict2 expects.
This creates the tokenized inputs and encoder outputs locally using a smaller T5 model.
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional

def generate_t5_tokens_and_embeddings(prompts: List[str], model_name: str = "google/flan-t5-xl"):
    """
    Generate T5 tokens and encoder embeddings for Cosmos Predict2.

    Uses Flan-T5-XL (3GB) as a substitute for T5-11B (45GB).
    Flan-T5 is actually better than vanilla T5 for instruction following.

    Args:
        prompts: List of text prompts for video generation
        model_name: T5 model to use (default: flan-t5-xl)

    Returns:
        Dict containing tokenized inputs and encoder outputs
    """

    try:
        from transformers import T5TokenizerFast, T5EncoderModel
    except ImportError:
        print("Error: transformers not installed")
        print("Install with: pip install transformers torch")
        return None

    print(f"Loading {model_name}...")
    print("This will download ~3GB on first run")

    # Load tokenizer and encoder model
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model = model.to('cuda')
        print("✅ Using GPU for encoding")
    else:
        print("⚠️ Using CPU (slower)")

    # Prepare storage for tokenized data
    tokenized_data = {}

    print(f"\nTokenizing {len(prompts)} prompts...")

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:60]}...")

            # Tokenize the prompt
            # T5 expects task prefix for better performance
            task_prompt = f"Generate video: {prompt}"

            inputs = tokenizer(
                task_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77  # Standard length for Cosmos
            )

            # Move to device
            if device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Get encoder outputs
            outputs = model(**inputs)

            # Extract the encoder hidden states
            encoder_hidden_states = outputs.last_hidden_state

            # Convert to CPU and numpy for storage
            hidden_states_np = encoder_hidden_states.cpu().numpy()
            input_ids_np = inputs['input_ids'].cpu().numpy()
            attention_mask_np = inputs['attention_mask'].cpu().numpy()

            # Store everything
            tokenized_data[f"prompt_{i}"] = {
                "text": prompt,
                "task_prompt": task_prompt,
                "input_ids": input_ids_np.tolist(),
                "attention_mask": attention_mask_np.tolist(),
                "encoder_hidden_states": hidden_states_np.tolist(),
                "shape": {
                    "batch_size": hidden_states_np.shape[0],
                    "sequence_length": hidden_states_np.shape[1],
                    "hidden_dim": hidden_states_np.shape[2]
                },
                "model": model_name
            }

            print(f"  ✅ Tokenized: shape {hidden_states_np.shape}")
            print(f"     Tokens: {len(input_ids_np[0])} tokens")

    return tokenized_data


def save_tokenized_data(tokenized_data: Dict, base_path: str = "t5_pretokenized"):
    """
    Save tokenized data in multiple formats for flexibility.
    """

    base_path = Path(base_path)

    # Save as JSON (human-readable but large)
    json_path = f"{base_path}.json"
    with open(json_path, 'w') as f:
        json.dump(tokenized_data, f, indent=2)
    print(f"\n✅ Saved JSON: {json_path}")

    # Save as pickle (faster loading, smaller)
    pkl_path = f"{base_path}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(tokenized_data, f)
    print(f"✅ Saved pickle: {pkl_path}")

    # Save as numpy arrays (most efficient for inference)
    npz_data = {}
    prompts_list = []

    for key, data in tokenized_data.items():
        npz_data[f"{key}_hidden"] = np.array(data['encoder_hidden_states'])
        npz_data[f"{key}_ids"] = np.array(data['input_ids'])
        npz_data[f"{key}_mask"] = np.array(data['attention_mask'])
        prompts_list.append(data['text'])

    npz_path = f"{base_path}.npz"
    np.savez_compressed(npz_path, **npz_data)

    # Save prompt list separately
    prompts_path = f"{base_path}_prompts.txt"
    with open(prompts_path, 'w') as f:
        for i, prompt in enumerate(prompts_list):
            f.write(f"{i}: {prompt}\n")

    print(f"✅ Saved numpy: {npz_path}")
    print(f"✅ Saved prompts list: {prompts_path}")

    return json_path, pkl_path, npz_path


def create_cosmos_compatible_embeddings():
    """
    Create embeddings specifically formatted for Cosmos Predict2 pipeline.
    """

    # Prompts for SO-101 paper manipulation task
    prompts = [
        "Robot arm picks up white paper and places it in red square target area",
        "Precise robotic manipulation of paper sheet to designated zone",
        "Gripper grasps paper edges and transfers to red square smoothly",
        "Close-up view of robot carefully handling paper material",
        "Multiple camera angles showing paper placement task",
        "Slow motion capture of gripper picking and placing paper",
        "Time-lapse of repeated paper manipulation cycles",
        "Robot learning to place paper accurately in target area",
        "Demonstration of fine motor control in paper handling",
        "Final precise positioning of paper in center of red square"
    ]

    print("=" * 70)
    print("Generating T5 Pre-tokenized Data for Cosmos Predict2")
    print("=" * 70)

    # Generate tokenized data
    tokenized_data = generate_t5_tokens_and_embeddings(prompts)

    if tokenized_data:
        # Save in multiple formats
        paths = save_tokenized_data(tokenized_data, "cosmos_t5_pretokenized")

        print("\n" + "=" * 70)
        print("Pre-tokenization Complete!")
        print("=" * 70)

        print("\nGenerated files:")
        for path in paths:
            if path:
                size = Path(path).stat().st_size / (1024 * 1024)
                print(f"  - {path} ({size:.2f} MB)")

        print("\n📝 Usage in Cosmos pipeline:")
        print("```python")
        print("# Load pre-tokenized data")
        print("import pickle")
        print("with open('cosmos_t5_pretokenized.pkl', 'rb') as f:")
        print("    pretokenized = pickle.load(f)")
        print("")
        print("# Get encoder hidden states for a prompt")
        print("hidden_states = pretokenized['prompt_0']['encoder_hidden_states']")
        print("# Use these instead of text prompt in Cosmos pipeline")
        print("```")

        return tokenized_data
    else:
        print("\n❌ Failed to generate tokenized data")
        return None


def load_and_verify():
    """
    Load and verify the saved tokenized data.
    """

    print("\n" + "=" * 70)
    print("Verifying Saved Data")
    print("=" * 70)

    # Load from pickle (fastest)
    with open('cosmos_t5_pretokenized.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"\nLoaded {len(data)} tokenized prompts")

    # Check first prompt
    first = data['prompt_0']
    hidden = np.array(first['encoder_hidden_states'])

    print(f"\nFirst prompt: '{first['text'][:60]}...'")
    print(f"Hidden states shape: {hidden.shape}")
    print(f"Model used: {first['model']}")
    print(f"Dimensions: {first['shape']}")

    # Load numpy version
    npz = np.load('cosmos_t5_pretokenized.npz')
    print(f"\nNumpy archive contains {len([k for k in npz.keys() if 'hidden' in k])} prompts")

    return data


if __name__ == "__main__":
    # Generate the pre-tokenized data
    tokenized_data = create_cosmos_compatible_embeddings()

    if tokenized_data:
        # Verify it loads correctly
        load_and_verify()

        print("\n✅ Pre-tokenization complete!")
        print("Use these files in your Cosmos Predict2 pipeline to avoid loading T5-11B")