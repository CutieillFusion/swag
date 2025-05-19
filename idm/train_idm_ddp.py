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
from dataclasses import dataclass, field
from typing import List
import torch.multiprocessing as mp
import random

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
    feature_channels: List[int] = field(default_factory=lambda: [16, 16, 16])
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
    epochs: int = 2
    seed: int = 42

    
def compute_class_weights(
    dataset: NumpyVideoDataset, num_classes: int, device: torch.device, min_class_weight: int = 1000
) -> torch.Tensor:
    all_labels = dataset.get_all_labels()

    existing_classes = np.unique(all_labels)

    class_weights_dict = dict(
        zip(
            existing_classes,
            compute_class_weight("balanced", classes=existing_classes, y=all_labels),
        )
    )

    class_weights = np.full(num_classes, min_class_weight, dtype=np.float32)
    for cls, weight in class_weights_dict.items():
        if weight > min_class_weight:
            class_weights[cls] = min_class_weight

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

def prepare_data_loaders(dataset: NumpyVideoDataset, train_ratio: float, world_size: int, rank: int, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
       
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_ratio, 1 - train_ratio]
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=seed
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_dataloader, val_dataloader

def train_idm(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
    training_config: TrainingConfig,
    job_id: str
) -> None:
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(training_config.epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        
        train_loss = 0.0
        train_samples = 0
        train_correct = 0
        
        for videos, labels in train_dataloader:
            videos: torch.Tensor = videos.to(device, non_blocking=True)
            labels: torch.Tensor = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits: torch.Tensor = model(videos)

            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)

            loss: torch.Tensor = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * videos.size(0)
            train_samples += videos.size(0)
            
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            
        train_metrics = torch.tensor([train_loss, train_samples, train_correct], device=device)
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        
        total_train_loss, total_train_samples, total_train_correct = train_metrics.tolist()
        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = total_train_correct / total_train_samples
        
        print_rank(f"Epoch {epoch + 1} training | Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f}", rank)

        dist.barrier(device_ids=[device.index])
        
        model.eval()
        val_dataloader.sampler.set_epoch(epoch)
        val_loss = 0.0
        val_samples = 0
        val_correct = 0
        
        for videos, labels in val_dataloader:
            with torch.no_grad():
                videos: torch.Tensor = videos.to(device, non_blocking=True)
                labels: torch.Tensor = labels.to(device, non_blocking=True)
                
                logits: torch.Tensor = model(videos)

                logits = logits.reshape(-1, logits.size(-1))
                labels = labels.reshape(-1)

                loss: torch.Tensor = criterion(logits, labels)

                val_loss += loss.item() * videos.size(0)
                val_samples += videos.size(0)
                
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
        
        val_metrics = torch.tensor([val_loss, val_samples, val_correct], device=device)
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
        total_val_loss, total_val_samples, total_val_correct = val_metrics.tolist()
        avg_val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples
        
        print_rank(f"Epoch {epoch + 1} validation | Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f}", rank)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if rank == 0:
                model_path = os.path.join("idm/models", job_id)
                os.makedirs(model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(model_path, f"idm_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= training_config.patience:
                print_rank(f"Early stopping triggered after {epoch + 1} epochs", rank)
                break
        
        dist.barrier(device_ids=[device.index])

def setup(local_rank: int, world_size: int, rank: int) -> None:
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup() -> None:
    dist.destroy_process_group()

def train(rank: int, local_rank: int, world_size: int, args: argparse.Namespace):
    setup(local_rank, world_size, rank)
    
    model_config, training_config = load_config(args.config)

    print_rank(model_config, rank)
    print_rank(training_config, rank)

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

    train_dataloader, val_dataloader = prepare_data_loaders(
        dataset=dataset,
        train_ratio=training_config.test_train_split,
        world_size=world_size,
        rank=rank,
        batch_size=training_config.batch_size,
        seed=training_config.seed
    )

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    
    start_time = time.time()

    num_classes = len(ACTION_SPACE)
    class_weights = compute_class_weights(dataset, num_classes, device, training_config.min_class_weight)
    print_rank(f"Class Weights {class_weights}", rank)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    train_idm(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        device,
        rank,
        training_config,
        args.job_id
    )

    end_time = time.time()
    total_training_time = end_time - start_time
    print_rank(f"Total training time: {total_training_time:.2f} seconds", rank)
    
    cleanup()

def main():
    args = parse_arguments()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank       = int(os.environ["RANK"])

    assert world_size >= 2, f"Need >=2 GPUs (got {world_size})"

    train(rank, local_rank, world_size, args)

if __name__ == "__main__":
    main()