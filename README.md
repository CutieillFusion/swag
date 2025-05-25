# SWAG: Video PreTraining (VPT) for NES Games

An implementation of OpenAI's Video PreTraining (VPT) process for NES game agents.

## Overview

This project recreates OpenAI's Video PreTraining (VPT) process from 2023, adapting it for NES games. VPT enables training AI agents to play video games by watching human gameplay demonstrations without requiring direct access to the environment reward function. This approach allows for learning complex behaviors that would be difficult to acquire through pure reinforcement learning.

The implementation follows a two-step process:
1. **Inverse Dynamics Model (IDM)**: Learns to predict human actions from gameplay observations
2. **Video PreTraining (VPT)**: Uses the IDM to label gameplay data for training a robust policy

## Architecture

The project uses a combination of convolutional layers for feature extraction and transformer blocks for temporal reasoning:

- **Spatial Feature Extractor**: 3D convolutions to process temporal information in gameplay frames
- **Visual Feature Extractor**: Stacked 2D convolutions to extract spatial features
- **Embedder**: Projects extracted features into a lower-dimensional embedding space
- **Transformer**: Processes the embedded sequence using normalized transformer blocks with causal attention
- **Classifier**: Maps transformer outputs to action probabilities

## Pipeline

The full VPT pipeline consists of the following steps:

1. **Data Preparation (IDM)**: Convert raw gameplay recordings to numpy format
   ```
   ./utils/scripts/1_convert_idm_raw_to_numpy.sh
   ```

2. **IDM Training**: Train the Inverse Dynamics Model on human gameplay
   ```
   ./utils/scripts/2_train_idm.sh [embedding_dim] [ff_dim] [transformer_blocks] [transformer_heads] [x] [y] [learning_rate] [weight_decay] [feature_channels]
   ```
   - DDP version available: `2_train_idm_ddp.sh`

3. **IDM Evaluation**: Evaluate the IDM's performance
   ```
   ./utils/scripts/3_evaluate_idm.sh
   ```

4. **Data Preparation (VPT)**: Convert raw gameplay for VPT training
   ```
   ./utils/scripts/4_convert_vpt_raw_to_numpy.sh
   ```

5. **Data Labeling**: Use the trained IDM to label additional gameplay data
   ```
   ./utils/scripts/5_label_with_idm.sh
   ```

6. **VPT Training**: Train the policy using labeled data
   ```
   ./utils/scripts/6_train_vpt.sh [embedding_dim] [ff_dim] [transformer_blocks] [transformer_heads] [x] [y] [learning_rate] [weight_decay] [feature_channels]
   ```
   - DDP version available: `6_train_vpt_ddp.sh`

7. **VPT Evaluation**: Evaluate the trained policy
   ```
   ./utils/scripts/7_evaluate_vpt.sh
   ```

8. **Fine-tuning**: Fine-tune the VPT model on specific games/tasks
   ```
   ./utils/scripts/8_finetune_vpt.sh [embedding_dim] [ff_dim] [transformer_blocks] [transformer_heads] [x] [y] [learning_rate] [weight_decay] [feature_channels]
   ```

## Container Setup

This project uses Docker and Singularity containers to ensure consistent environments across different computing systems.

### Docker Container

The project uses a Docker container available at:
```
norquistdylan/nes-25:latest
```

To pull the Docker image:
```bash
docker pull norquistdylan/nes-25:latest
```

#### Building Your Own Docker Image

If you need to customize the Docker image, you can build it from the provided Dockerfile:

1. **Generate requirements file** (if your dependencies have changed):
   ```bash
   cd /data/ai_club/nes_2025/swag
   ./utils/containers/get_requirements.sh
   ```
   This script will generate a `requirements.txt` file based on your current Python environment.

2. **Build the Docker image**:
   ```bash
   cd /data/ai_club/nes_2025/swag
   docker build -f utils/containers/Dockerfile -t your-username/nes-25:latest .
   ```

3. **Push to Docker Hub**:
   ```bash
   docker login
   docker push your-username/nes-25:latest
   ```

### Singularity Container

For HPC environments, a Singularity container is used. The container is built from the Docker image:

1. **Build the Singularity container**:
   ```bash
   cd /data/ai_club/nes_2025/swag
   ./utils/containers/build_singularity_container.sh
   ```

   This script builds the Singularity container at `utils/containers/container.sif` from the Docker image.

2. **Using the Singularity container**:
   The training scripts automatically use the Singularity container with the correct configuration. For example:
   ```bash
   singularity exec \
       --env PYTHONPATH=$PYTHONPATH \
       --nv \
       -B /path/to/swag:/path/to/swag \
       utils/containers/container.sif \
       python your_script.py
   ```

3. **Testing the Singularity container**:
   The project includes a script to verify that the container has all required dependencies installed with correct versions:
   ```bash
   cd /data/ai_club/nes_2025/swag
   ./utils/containers/test_singularity_container.sh
   ```
   
   This script executes a Python test that checks all packages from `requirements.txt` and verifies they are installed in the container with the correct versions. You'll see a summary report showing:
   - ✅ Correctly installed packages
   - ❌ Missing packages 
   - ⚠️ Packages with version mismatches

   Always run this test after building or updating a container to ensure compatibility.

#### Building a Custom Singularity Container

If you built a custom Docker image, update the `build_singularity_container.sh` script:

1. **Edit the build script**:
   ```bash
   # Open the script
   nano utils/containers/build_singularity_container.sh
   
   # Change the DOCKER_IMAGE variable to your custom image
   # DOCKER_IMAGE="your-username/nes-25:latest"
   ```

2. **Build your custom Singularity container**:
   ```bash
   ./utils/containers/build_singularity_container.sh
   ```

3. **Test your custom container**:
   ```bash
   ./utils/containers/test_singularity_container.sh
   ```

## Configuration

The project uses YAML configuration files to manage model architecture and training parameters:

- **vpt.yaml**: Configuration for VPT models
- **idm.yaml**: Configuration for IDM models

Example configuration:
```yaml
model:
  embedding_dim: 1024
  ff_dim: 4096
  transformer_heads: 8
  transformer_blocks: 4
  dimensions:
    x: 256
    y: 240
  feature_channels: [64, 128, 128]
  spatial_channels: 64
  sequence_length: 64

training:
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.0001
  patience: 10
  stride: 4
  data_dir: "vpt/data/numpy"
  test_train_split: 0.9
  min_class_weight: 1000
  epochs: 100
```

## Requirements

The project has been tested on Python 3.8+ and requires several dependencies listed in `requirements.txt`. Key dependencies include:

- PyTorch
- Gym Super Mario Bros
- NumPy
- OpenCV

## High-Performance Computing Integration

The scripts are designed to integrate with SLURM-based HPC environments, with configurable parameters for:
- GPU allocation
- Memory requirements
- CPU cores
- Job time limits

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Build the Singularity container (if using HPC)
4. Test the container using `./utils/containers/test_singularity_container.sh`
5. Follow the pipeline steps sequentially, starting with data preparation

## References

This work is based on OpenAI's Video PreTraining methodology:
- [OpenAI VPT Paper](https://arxiv.org/abs/2206.11795) (2022)