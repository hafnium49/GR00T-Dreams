# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Code Quality and Testing
```bash
# Run all checks (formatting, linting, tests)
make run-checks

# Format code
make format
# OR individually:
isort .
black .

# Linting
ruff check .

# Run tests
pytest -v --color=yes tests/

# Run specific test
pytest -v tests/test_dataset.py

# Build package
make build
```

### Development Setup
```bash
# Install in development mode
pip install -U pip setuptools wheel
pip install -e .[dev]

# Create conda environment (recommended)
conda create -n isaac-gr00t python=3.10
conda activate isaac-gr00t
```

## Core Training Scripts

### IDM (Inverse Dynamics Model) Training
```bash
# Train IDM model
PYTHONPATH=. torchrun scripts/idm_training.py --dataset-path <path> --embodiment_tag <tag>

# Example for GR1 embodiment
PYTHONPATH=. torchrun scripts/idm_training.py --dataset-path demo_data/robot_sim.PickNPlace/ --embodiment_tag gr1
```

### GR00T N1 Fine-tuning
```bash
# Fine-tune GR00T N1 model
PYTHONPATH=. torchrun scripts/gr00t_finetune.py --dataset-path <path> --data-config <config>
```

### Evaluation and Inference
```bash
# Evaluate policy
python scripts/eval_policy.py

# Run inference service
python scripts/inference_service.py
```

## Video Processing Pipeline

### Convert Directory Structure (Step 3.1)
```bash
python IDM_dump/convert_directory.py \
    --input_dir "${COSMOS_PREDICT2_OUTPUT_DIR}" \
    --output_dir "results/dream_gen_benchmark/cosmos_predict2_14b_gr1_object_step3"
```

### Extract Actions (Step 3.2)
Processing scripts located in `IDM_dump/scripts/preprocess/` for different embodiments:
- `franka`: Franka Emika Panda Robot Arm
- `gr1`: Fourier GR1 Humanoid Robot
- `so100`: SO-100 Robot Arm
- `robocasa`: RoboCasa (Simulation)

## DreamGenBench Evaluation

### Success Rate Evaluation
```bash
# Using Qwen2.5-VL
python -m dreamgenbench.eval_sr_qwen_whole \
    --video_dir "$video_dir" \
    --output_csv "$csv_path" \
    --device "$device"

# Using GPT-4o
python -m dreamgenbench.eval_sr_gpt4o_whole \
    --video_dir "$video_dir" \
    --output_csv "$csv_path" \
    --api_key "$api_key"

# Physical Alignment (PA Score I)
python -m dreamgenbench.eval_qwen_pa \
    --video_dir "$video_dir" \
    --output_csv "$csv_path" \
    --device "$device"
```

## Architecture Overview

### Package Structure
- **gr00t/**: Main package containing core functionality
  - **data/**: Data handling, schemas, and transformations
    - `dataset.py`: LeRobot dataset implementations
    - `schema.py`: Data schemas and embodiment definitions
    - `transform/`: Data transformation utilities
  - **model/**: Neural network models and architectures
    - `gr00t_n1.py`: Main GR00T N1 model implementation
    - `idm.py`: Inverse Dynamics Model
    - `policy.py`: Policy model implementations
    - `action_head/`: Action prediction heads
    - `backbone/`: Model backbone architectures
  - **experiment/**: Training configurations and runners
    - `data_config.py`: Data configurations for GR00T training
    - `data_config_idm.py`: Data configurations for IDM training
    - `runner.py`: Training runner for GR00T
    - `runner_idm.py`: Training runner for IDM
  - **eval/**: Evaluation utilities and wrappers
  - **utils/**: Shared utilities

### Key Scripts
- **scripts/**: Main training and evaluation entry points
- **IDM_dump/**: Video processing and action extraction pipeline
- **dreamgenbench/**: DreamGenBench evaluation scripts
- **tests/**: Unit tests

### Supported Embodiments
The codebase supports multiple robot embodiments through configuration:
- **GR1**: Fourier GR1 Humanoid Robot
- **Franka**: Franka Emika Panda Robot Arm
- **SO-100**: SO-100 Robot Arm
- **RoboCasa**: RoboCasa Simulation Environment

### Data Pipeline
1. **Video World Model Fine-tuning**: Using cosmos-predict2
2. **Synthetic Video Generation**: Generate robot videos with fine-tuned models
3. **Action Extraction**: Extract actions using IDM models to LeRobot format
4. **Policy Training**: Fine-tune GR00T N1 on extracted data
5. **Evaluation**: Assess performance using DreamGenBench metrics

## Configuration Management
- Uses Hydra for configuration management
- Data configurations stored in `gr00t/experiment/data_config*.py`
- Support for different embodiments through `embodiment_tag` parameter
- Model configurations include batch size, learning rate, and training steps

## Dependencies
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **LeRobot**: Robot learning data format compatibility
- **Hydra**: Configuration management
- **Wandb**: Experiment tracking and logging
- **Various CV libraries**: OpenCV, albumentations, decord for video processing

Always use `PYTHONPATH=.` when running training scripts to ensure proper module imports.