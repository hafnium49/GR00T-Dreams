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

# Alternative: Use automated setup script
bash setup_environment.sh

# For Cosmos Predict2 environment setup
bash setup_cosmos_env.sh
```

## Core Training Scripts

### IDM (Inverse Dynamics Model) Training
```bash
# Train IDM model
PYTHONPATH=. torchrun scripts/idm_training.py --dataset-path <path> --embodiment_tag <tag>

# Example for GR1 embodiment
PYTHONPATH=. torchrun scripts/idm_training.py --dataset-path demo_data/robot_sim.PickNPlace/ --embodiment_tag gr1

# Example for RoboCasa
PYTHONPATH=. torchrun scripts/idm_training.py --dataset-path <path> --data-config single_panda_gripper --embodiment_tag robocasa_panda_omron
```

### GR00T N1 Fine-tuning
```bash
# Fine-tune GR00T N1 model
PYTHONPATH=. torchrun scripts/gr00t_finetune.py --dataset-path <path> --data-config <config>

# Available data configs (defined in gr00t/experiment/data_config.py):
# - gr1_arms_only, gr1_full_body, gr1_arms_waist
# - franka_left_arm, franka_right_arm
# - so100
# - single_panda_gripper, bimanual_panda_gripper, bimanual_panda_hand

# Quick fine-tuning for specific embodiments (from IDM_dump/scripts/finetune/):
bash IDM_dump/scripts/finetune/gr1.sh
bash IDM_dump/scripts/finetune/franka.sh
bash IDM_dump/scripts/finetune/so100.sh
bash IDM_dump/scripts/finetune/robocasa.sh
```

### Evaluation and Inference
```bash
# Evaluate policy
python scripts/eval_policy.py

# Run inference service
python scripts/inference_service.py

# Load and validate dataset
python scripts/load_dataset.py --dataset-path <path>
```

## Complete DreamGen Pipeline Workflow

The full pipeline for generating synthetic training data and fine-tuning robots:

### 1. Fine-tune Video World Model
Use Cosmos Predict2 to fine-tune on robot demonstrations. See external documentation:
- [Cosmos Predict2 Training Guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world_gr00t.md)

### 2. Generate Synthetic Videos
Generate robot videos using fine-tuned Cosmos Predict2 model.

### 3. Process Videos to Training Data

#### 3.1 Convert Directory Structure
```bash
python IDM_dump/convert_directory.py \
    --input_dir "${COSMOS_PREDICT2_OUTPUT_DIR}" \
    --output_dir "results/dream_gen_benchmark/cosmos_predict2_14b_gr1_object_step3"
```

#### 3.2 Extract Actions
Processing scripts located in `IDM_dump/scripts/preprocess/` for different embodiments:
- `franka.sh`: Franka Emika Panda Robot Arm
- `gr1.sh`: Fourier GR1 Humanoid Robot
- `so100.sh`: SO-100 Robot Arm
- `robocasa.sh`: RoboCasa (Simulation)
- `paper_return.sh`: Paper return dataset

Each script runs these steps in sequence:
1. `split_video_instruction.py`: Split videos by task/instruction
2. `preprocess_video.py`: Extract frames from videos
3. `raw_to_lerobot.py`: Convert to LeRobot format
4. `dump_idm_actions.py`: Extract actions using pre-trained IDM model

### 4. Fine-tune GR00T N1 Policy
Use the extracted action data to fine-tune the policy model (see GR00T N1 Fine-tuning section).

### 5. Evaluate Performance
Use DreamGenBench metrics to evaluate the fine-tuned model (see DreamGenBench Evaluation section).

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
    - `dataset.py`: LeRobot dataset implementations (LeRobotSingleDataset, LeRobotDataset)
    - `schema.py`: Data schemas and embodiment definitions
    - `embodiment_tags.py`: Supported robot embodiments enum (GR1, FRANKA, SO100, ROBOCASA, NEW_EMBODIMENT)
    - `transform/`: Data transformation utilities (video, state_action, concat transforms)
  - **model/**: Neural network models and architectures
    - `gr00t_n1.py`: Main GR00T N1 vision-language-action model
    - `idm.py`: Inverse Dynamics Model for action extraction
    - `policy.py`: Policy model implementations
    - `transforms.py`: GR00T model transforms
    - `transforms_idm.py`: IDM model transforms
    - `action_head/`: Action prediction heads (flow matching, DiT)
    - `backbone/`: Model backbone architectures (Eagle, Eagle2)
  - **experiment/**: Training configurations and runners
    - `data_config.py`: Data configurations for GR00T training
    - `data_config_idm.py`: Data configurations for IDM training
    - `runner.py`: Training runner for GR00T
    - `runner_idm.py`: Training runner for IDM
    - `trainer.py`: Custom DualBrainTrainer
  - **eval/**: Evaluation utilities and wrappers
    - `service.py`: Inference service implementation
    - `robot.py`: Robot control interface
    - `wrappers/`: Environment wrappers for evaluation
  - **utils/**: Shared utilities

### Key Scripts
- **scripts/**: Main training and evaluation entry points
- **IDM_dump/**: Video processing and action extraction pipeline
- **dreamgenbench/**: DreamGenBench evaluation scripts
- **tests/**: Unit tests

### Supported Embodiments
The codebase supports multiple robot embodiments through configuration:
- **GR1**: Fourier GR1 Humanoid Robot (arms, hands, waist)
- **Franka**: Franka Emika Panda Robot Arm (7-DOF)
- **SO-100**: SO-100 Robot Arm (6-DOF)
- **RoboCasa**: RoboCasa Simulation Environment
- **NEW_EMBODIMENT**: Template for custom robots

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
- Metadata files in `IDM_dump/global_metadata/{embodiment}/`:
  - `modality.json`: State/action space definition
  - `stats.json`: Normalization statistics

## Pre-trained Models and Checkpoints

### IDM Models (HuggingFace)
Pre-trained IDM models available for action extraction:
- `seonghyeonye/IDM_gr1`: GR1 humanoid robot
- `seonghyeonye/IDM_franka`: Franka Panda arm
- `seonghyeonye/IDM_so100`: SO-100 arm
- `seonghyeonye/IDM_robocasa`: RoboCasa simulation

These are used automatically by the preprocessing scripts in `IDM_dump/scripts/preprocess/`.

## Key Classes and Interfaces

### Dataset Classes
- `LeRobotSingleDataset`: Single trajectory dataset
- `LeRobotDataset`: Multi-trajectory dataset with language instructions
- Supports multi-modal data: video, state, action, language

### Model Components
- **GR00T N1**: Vision-language-action model with:
  - Eagle/Eagle2 visual encoder
  - Language model (LLaMA/Qwen based)
  - Flow matching action head
- **IDM**: Inverse dynamics model for action extraction
- **PEFT Support**: LoRA for efficient fine-tuning

### Training Infrastructure
- Uses HuggingFace Trainer with custom callbacks
- Distributed training via torchrun
- Wandb integration for experiment tracking
- Checkpoint management with best model selection

## Dependencies
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **LeRobot**: Robot learning data format compatibility
- **Hydra**: Configuration management
- **Wandb**: Experiment tracking and logging
- **Various CV libraries**: OpenCV, albumentations, decord for video processing

## Important Notes
- Always use `PYTHONPATH=.` when running training scripts to ensure proper module imports
- LeRobot format uses Parquet for tabular data and MP4 for videos
- Support for multi-GPU training via torchrun
- Custom embodiments require modality.json and stats.json in `IDM_dump/global_metadata/{embodiment}/`
- When adding new embodiments, also update `gr00t/data/embodiment_tags.py` enum
- Default batch sizes and learning rates are configured per embodiment in training scripts
- Recommended fine-tuning: boost batch size to max GPU capacity, train for 20k steps