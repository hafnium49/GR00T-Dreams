#!/usr/bin/env python3
"""
Quick environment test script for GR00T Dreams
Run this after activating the conda environment to verify everything works
"""

import sys
import subprocess
from typing import List, Tuple

def test_import(module_name: str, display_name: str = None) -> Tuple[bool, str]:
    """Test importing a module and return success status with version info"""
    if display_name is None:
        display_name = module_name
    
    try:
        if module_name == "torch":
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "Not available"
            return True, f"PyTorch {version} (CUDA: {cuda_version})"
        elif module_name == "torchvision":
            import torchvision
            return True, f"TorchVision {torchvision.__version__}"
        elif module_name == "transformers":
            import transformers
            return True, f"Transformers {transformers.__version__}"
        elif module_name == "cv2":
            import cv2
            return True, f"OpenCV {cv2.__version__}"
        elif module_name == "numpy":
            import numpy as np
            return True, f"NumPy {np.__version__}"
        elif module_name == "pandas":
            import pandas as pd
            return True, f"Pandas {pd.__version__}"
        elif module_name == "wandb":
            import wandb
            return True, f"W&B {wandb.__version__}"
        elif module_name == "einops":
            import einops
            return True, f"Einops {einops.__version__}"
        else:
            __import__(module_name)
            return True, "✅ Imported successfully"
            
    except ImportError as e:
        return False, f"❌ Import failed: {e}"
    except Exception as e:
        return False, f"❌ Error: {e}"

def main():
    print("🤖 GR00T Dreams Environment Test")
    print("=" * 50)
    
    # Test core packages
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("transformers", "Transformers"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("wandb", "Weights & Biases"),
        ("einops", "Einops"),
        ("pydantic", "Pydantic"),
        ("yaml", "PyYAML"),
        ("h5py", "HDF5"),
        ("imageio", "ImageIO"),
    ]
    
    results = []
    max_name_len = max(len(display_name) for _, display_name in packages)
    
    for module, display_name in packages:
        success, info = test_import(module, display_name)
        status = "✅" if success else "❌"
        print(f"{display_name:<{max_name_len}} : {status} {info}")
        results.append(success)
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready for GR00T Dreams.")
        
        # Additional CUDA info
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 CUDA Device: {torch.cuda.get_device_name(0)}")
                print(f"🔥 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except:
            pass
            
    else:
        print("⚠️  Some packages failed to import. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())