import argparse
from sklearn.utils.class_weight import compute_class_weight
import torch
import os
import time
from idm import IDM
from torch import nn
import torch.distributed as dist
from ddp_dataloader import NumpyVideoDataset, get_all_videos
from actions import ACTION_SPACE
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import yaml
from dataclasses import dataclass
from typing import List
import torch.multiprocessing as mp

def print_rank(message: str, rank: int) -> None:
    if rank == 0:
        print(message)

@dataclass
class ModelConfig:
    embedding_dim: int = 256
    ff_dim: int = 256
    transformer_heads: int = 4
    transformer_blocks: int = 1
    x: int = 32
    y: int = 30
    feature_channels: List[int] = [16, 16, 16]
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    batch_size: int = 4
    stride: int = 16
    spatial_channels: int = 64
    sequence_length: int = 64


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    patience: int = 10
    stride: int = 1
    data_dir: str = "idm/data/numpy"
    test_train_split: float = 0.8
    min_class_weight: int = 1000

def compute_class_weights(
    dataloader: NumpyVideoDataset, num_classes: int, device: torch.device, min_class_weight: int = 1000
) -> torch.Tensor:
    all_labels = []

    for _, labels in dataloader:
        all_labels.extend(labels.cpu().numpy().flatten())

    existing_classes = np.unique(all_labels)

    class_weights_dict = dict(
        zip(
            existing_classes,
            compute_class_weight("balanced", classes=existing_classes, y=all_labels),
        )
    )

    class_weights = np.full(num_classes, min_class_weight, dtype=np.float32)
    for cls, weight in class_weights_dict.items():
        class_weights[cls] = weight

    return torch.tensor(class_weights, dtype=torch.float).to(device)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Argument Parser for Model Configuration"
    )
    parser.add_argument(
        "--config", type=str, default="idm.yaml", help="Path to config file"
    )
    parser.add_argument("--job_id", type=str, required=True, help="Job identifier")
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> tuple[ModelConfig, TrainingConfig]:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    model_config = ModelConfig()
    if "model" in config_dict:
        for key, value in config_dict["model"].items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            elif key == "dimensions" and isinstance(value, dict):
                if "x" in value:
                    model_config.x = value["x"]
                if "y" in value:
                    model_config.y = value["y"]
    
    training_config = TrainingConfig()
    if "training" in config_dict:
        for key, value in config_dict["training"].items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
    
    return model_config, training_config

def setup(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup() -> None:
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    
    model_config, training_config = load_config(args.config)

    model_path = os.path.join("idm/models", str(args.job_id))
    os.makedirs(model_path, exist_ok=True)

    input_dims = (model_config.sequence_length, 3, model_config.y, model_config.x)
    print_rank(f"Input Dims {input_dims}", rank)
    print_rank(f"Feature Channels {model_config.feature_channels}", rank)
    print_rank(f"Transformer Blocks {model_config.transformer_blocks}", rank)
    print_rank(f"Transformer Heads {model_config.transformer_heads}", rank)
    print_rank(f"FeedForward Dimension {model_config.ff_dim}", rank)
    print_rank(f"Embedding Dimension {model_config.embedding_dim}", rank)
    print_rank(f"Learning Rate {training_config.learning_rate}", rank)
    print_rank(f"Weight Decay {training_config.weight_decay}", rank)
    print_rank(f"Spatial Channels {model_config.spatial_channels}", rank)

    model = IDM(
        n_actions=len(ACTION_SPACE),
        input_dim=input_dims,
        feature_channels=model_config.feature_channels,
        transformer_blocks=model_config.transformer_blocks,
        transformer_heads=model_config.transformer_heads,
        ff_dim=model_config.ff_dim,
        embedding_dim=model_config.embedding_dim,
        spatial_channels=model_config.spatial_channels,
    )
    
    if rank == 0:
        model.print_model_parameters()

    dir_path = training_config.data_dir
    ids = get_all_videos(dir_path)
    print_rank(f"Found {len(ids)} video directories", rank)

    dataset = NumpyVideoDataset(
        video_ids=ids,
        data_dir=dir_path,
        sequence_length=model_config.sequence_length,
        stride=training_config.stride,
        has_labels=True,
        filter_thresholds={
            0: 3,
            2: 3,
        },
        rank=rank,
        is_vpt=False,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=training_config.batch_size, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    start_time = time.time()

    num_classes = len(ACTION_SPACE)
    class_weights = compute_class_weights(dataset, num_classes, device, training_config.min_class_weight)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    end_time = time.time()
    total_training_time = end_time - start_time
    print_rank(f"Total training time: {total_training_time:.2f} seconds", rank)
    
    cleanup()


def main():
    args = parse_arguments()
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    world_size = torch.cuda.device_count()
    assert world_size >= 2, "Need at least 2 GPUs for DDP training"
    
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
