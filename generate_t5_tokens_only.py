#!/usr/bin/env python3
"""
Generate T5 tokens (input_ids and attention_mask) for Cosmos Predict2.
This only uses the T5 tokenizer, NOT the model - no large downloads required.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import T5TokenizerFast


def generate_t5_tokens(
    prompts: List[str],
    model_name: str = "google-t5/t5-11b",
    max_length: int = 77,  # Cosmos Predict2 default
) -> Dict[str, Any]:
    """
    Generate T5 tokens for Cosmos Predict2 using only the tokenizer.

    Args:
        prompts: List of text prompts to tokenize
        model_name: T5 model name (only tokenizer will be downloaded)
        max_length: Maximum token length

    Returns:
        Dictionary containing input_ids and attention_mask
    """
    print(f"Loading T5 tokenizer from {model_name}...")
    # Only download the tokenizer (small files), not the model
    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    results = {
        "prompts": prompts,
        "tokenizer_name": model_name,
        "max_length": max_length,
        "tokenized": []
    }

    for prompt in prompts:
        print(f"Tokenizing: {prompt[:50]}...")

        # Tokenize the text
        tokens = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Store tokenized data
        tokenized_data = {
            "prompt": prompt,
            "input_ids": tokens["input_ids"],  # Shape: [1, max_length]
            "attention_mask": tokens["attention_mask"],  # Shape: [1, max_length]
            # Note: We don't have encoder_hidden_states without running the model
            # Cosmos will need to run the T5 encoder with these tokens
        }

        results["tokenized"].append(tokenized_data)

        # Print token info
        num_tokens = tokens["attention_mask"].sum().item()
        print(f"  - Tokens used: {num_tokens}/{max_length}")
        print(f"  - Input IDs shape: {tokens['input_ids'].shape}")

    return results


def save_tokenized_data(data: Dict[str, Any], output_dir: str = "."):
    """Save tokenized data in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle (preserves torch tensors)
    pickle_path = output_path / "cosmos_t5_tokens.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved tokenized data to {pickle_path}")

    # Save as JSON (converts tensors to lists for readability)
    json_data = {
        "prompts": data["prompts"],
        "tokenizer_name": data["tokenizer_name"],
        "max_length": data["max_length"],
        "tokenized": []
    }

    for item in data["tokenized"]:
        json_data["tokenized"].append({
            "prompt": item["prompt"],
            "input_ids": item["input_ids"].tolist(),
            "attention_mask": item["attention_mask"].tolist()
        })

    json_path = output_path / "cosmos_t5_tokens.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved tokenized data to {json_path}")

    return pickle_path, json_path


def main():
    # Example prompts for paper manipulation task
    prompts = [
        "The robot picks up a piece of paper from the table",
        "The robot folds the paper in half",
        "The robot places the folded paper back on the table",
        "The robot arm reaches for a sheet of paper",
        "The robot grasps the paper with its gripper",
        "The robot lifts the paper from the surface",
        "The robot moves the paper to a new location",
        "The robot releases the paper",
        "A robotic arm performing paper manipulation task",
        "Robot handling paper with precise movements"
    ]

    # Generate tokens (only uses tokenizer, no model download)
    tokenized_data = generate_t5_tokens(prompts)

    # Save the tokenized data
    pickle_path, json_path = save_tokenized_data(tokenized_data)

    print("\n" + "="*50)
    print("Tokenization complete!")
    print(f"Generated tokens for {len(prompts)} prompts")
    print(f"Files saved:")
    print(f"  - {pickle_path} (for Python/PyTorch)")
    print(f"  - {json_path} (for inspection)")
    print("\nNote: These are just tokens (input_ids and attention_mask).")
    print("The T5 encoder still needs to process these tokens to generate embeddings.")
    print("This avoids downloading the 45GB model just for tokenization.")


if __name__ == "__main__":
    main()