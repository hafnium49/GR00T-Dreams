#!/usr/bin/env python
"""Quick test to check Cosmos Predict2 setup"""

import sys
sys.path.insert(0, '/home/hafnium/cosmos-predict2')

print("=== Quick Cosmos Predict2 Setup Check ===\n")

# Check PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

# Check key dependencies
dependencies = [
    'cv2',
    'numpy',
    'PIL',
    'imageio',
    'hydra',
    'omegaconf',
    'loguru',
    'transformers',
    'diffusers',
]

print("\n=== Dependency Check ===")
for dep in dependencies:
    try:
        if dep == 'cv2':
            import cv2
            print(f"✅ OpenCV: {cv2.__version__}")
        elif dep == 'PIL':
            from PIL import Image
            print(f"✅ Pillow: installed")
        else:
            __import__(dep)
            print(f"✅ {dep}: installed")
    except ImportError:
        print(f"❌ {dep}: not installed")

# Try to import cosmos modules
print("\n=== Cosmos Modules ===")
try:
    import imaginaire
    print("✅ imaginaire: installed")
except ImportError:
    print("❌ imaginaire: not installed")

try:
    import cosmos_predict2
    print("✅ cosmos_predict2: installed")
except ImportError:
    print("❌ cosmos_predict2: not installed")

print("\n=== Summary ===")
print("If all modules show ✅, you're ready to run inference!")
print("To run inference: conda run -n cosmos python test_cosmos_inference.py")