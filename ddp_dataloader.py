import os
import re
import random
import torch
import numpy as np
import time
from actions import convert_int_to_action, action_meanings
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

def print_rank(message: str, rank: int) -> None:
    if rank == 0:
        print(message)

class NumpyVideoDataset:
    def __init__(
        self,
        video_ids: list[str],
        data_dir: str,
        sequence_length: int = 64,
        stride: int = 16,
        has_labels: bool = True,
        filter_thresholds: dict = None,
        rank: int = 0,
        is_vpt: bool = False,
    ):
        self.video_ids = video_ids
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.has_labels = has_labels
        self.filter_thresholds = filter_thresholds or {}
        self.rank = rank
        self.is_vpt = is_vpt

        self.video_info = []
        self.sequence_indices = []

        start_time = time.time()
        self._gather_video_metadata()
        print_rank(f"Total videos processed: {len(self.video_info)}", self.rank)
        
        self._build_indices()
        print_rank(f"Total sequences: {len(self.sequence_indices)}", self.rank)
        
        if self.has_labels:
            for label, threshold in self.filter_thresholds.items():
                self._filter_sequences(label, threshold)
                print_rank(f"Total sequences after filtering label {action_meanings[label]}: {len(self.sequence_indices)}", self.rank)

        end_time = time.time()
        print_rank(f"Total time for dataset initialization: {end_time - start_time:.2f} seconds", self.rank)

    def _gather_video_metadata(self) -> None:
        for vid_id in self.video_ids:
            chunk_files = []
            video_dir = os.path.join(self.data_dir, str(vid_id))
            for fname in os.listdir(video_dir):
                if "frames_" in fname and fname.endswith(".npy"):
                    match = re.match(r"frames_(\d+)_(\d+)\.npy", fname)
                    if match:
                        start = int(match.group(1))
                        end = int(match.group(2))
                        full_path = os.path.join(video_dir, fname)
                        chunk_files.append((start, end, full_path))
            
            chunk_files.sort(key=lambda x: x[0])
            if len(chunk_files) == 0:
                print_rank(f"No chunk files found for video {vid_id}. Skipping.", self.rank)
                continue

            labels = None
            if self.has_labels:
                label_path = os.path.join(video_dir, "labels.txt")
                try:
                    with open(label_path, "r") as f:
                        labels = f.read().split()
                        labels = [convert_int_to_action(int(label, 2)) for label in labels]
                        labels = np.array(labels, dtype=np.int64)
                except Exception as e:
                    print_rank(f"Error reading label file {label_path}: {e}", self.rank)
                    continue
            
            total_frames = chunk_files[-1][1]
            if self.has_labels and len(labels) != total_frames:
                print_rank(f"Warning: For video {vid_id}, total_frames={total_frames}, but label count={len(labels)}", self.rank)
            
            self.video_info.append({
                "video_id": vid_id,
                "chunks": chunk_files,
                "labels": labels,
                "num_frames": min(total_frames, len(labels) if self.has_labels else total_frames),
            })

    def _build_indices(self) -> None:
        for vid_idx, info in enumerate(self.video_info):
            n_frames = info["num_frames"]
            end = n_frames - self.sequence_length if self.is_vpt else n_frames - self.sequence_length + 1
            for start in range(0, end, self.stride):
                self.sequence_indices.append((vid_idx, start))

    def _filter_sequences(self, label_val, threshold) -> None:
        if not self.has_labels:
            return
        
        filtered = []
        for vid_idx, start in self.sequence_indices:
            info = self.video_info[vid_idx]
            seq_labels = info["labels"][start:start + self.sequence_length]
            if not self._has_consecutive(seq_labels, label_val, threshold):
                filtered.append((vid_idx, start))
        
        self.sequence_indices = filtered

    @staticmethod
    def _has_consecutive(seq, label_val, threshold):
        count = 0
        for label in seq:
            if label == label_val:
                count += 1
                if count >= threshold:
                    return True
            else:
                count = 0
        return False

    def get_all_labels(self) -> list:
        if not self.has_labels:
            return []
            
        all_labels = []
        for vid_idx, start in self.sequence_indices:
            info = self.video_info[vid_idx]
            end = start + self.sequence_length
            seq_labels = info["labels"][(start + 1) if self.is_vpt else start:
                                       (end + 1) if self.is_vpt else end]
            all_labels.extend(seq_labels)
            
        return all_labels

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        vid_idx, start_frame = self.sequence_indices[idx]
        info = self.video_info[vid_idx]
        end_frame = start_frame + self.sequence_length
        
        clip_frames = self._load_clip_frames(info["chunks"], start_frame, end_frame)
        
        label_tensor = None
        if self.has_labels:
            clip_labels = info["labels"][(start_frame + 1) if self.is_vpt else start_frame:
                                         (end_frame + 1) if self.is_vpt else end_frame]
            label_tensor = torch.tensor(clip_labels, dtype=torch.long)

        return clip_frames, label_tensor

    def _load_clip_frames(
        self, chunks: list[tuple[int, int, str]], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        frames_needed = end_frame - start_frame
        current_frame = start_frame
        frames_collected = []
        chunk_idx = 0
        
        while frames_needed > 0 and chunk_idx < len(chunks):
            chunk_start, chunk_end, chunk_path = chunks[chunk_idx]
            
            if chunk_end <= current_frame:
                chunk_idx += 1
                continue
            
            if chunk_start >= end_frame:
                break

            chunk_array = np.load(chunk_path, mmap_mode="r+")
            chunk_tensor = torch.from_numpy(chunk_array).float()
            chunk_tensor = chunk_tensor / 255.0

            offset_in_chunk = current_frame - chunk_start
            chunk_frames_avail = chunk_end - current_frame
            take = min(chunk_frames_avail, frames_needed)
            portion = chunk_tensor[offset_in_chunk:offset_in_chunk + take]
            frames_collected.append(portion)
            
            frames_needed -= take
            current_frame += take
            
            if current_frame >= chunk_end:
                chunk_idx += 1
        
        all_frames = torch.cat(frames_collected, dim=0)
        return all_frames


def get_all_videos(directory_path: str, has_labels: bool = True) -> list[str]:
    dirs = []
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    if not has_labels or os.path.exists(os.path.join(entry, "labels.txt")):
                        dirs.append(entry.name)
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
    except NotADirectoryError:
        print(f"Error: '{directory_path}' is not a directory.")

    random.shuffle(dirs)
    return dirs

def setup(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup() -> None:
    dist.destroy_process_group()

def run_worker(rank, world_size):
    setup(rank, world_size)
    
    dir_path = "idm/data/numpy"
    ids = get_all_videos(dir_path, has_labels=True)
    print_rank(f"Found {len(ids)} video directories", rank)

    video_dataset = NumpyVideoDataset(
        video_ids=ids,
        data_dir=dir_path,
        sequence_length=64,
        stride=32,
        has_labels=True,
        filter_thresholds={
            0: 3,
            2: 3,
        },
        rank=rank,
        is_vpt=False,
    )

    sampler = DistributedSampler(
        video_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        video_dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print_rank(f"Dataset size: {len(video_dataset)}, Dataloader batches: {len(dataloader)}", rank)
    
    for epoch in range(1, 4):
        start_time = time.time()
        total_samples = 0
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos, labels = videos.to(torch.cuda.current_device(), non_blocking=True), labels.to(torch.cuda.current_device(), non_blocking=True)
            batch_size = videos.size(0)
            total_samples += batch_size
        end_time = time.time()
        
        print_rank(f"Epoch {epoch}: Total time for {len(dataloader)} batches ({total_samples} samples) is {end_time - start_time:.2f} seconds", rank)
        print_rank(f"Epoch {epoch}: Average time per batch: {(end_time - start_time) / len(dataloader):.4f} seconds", rank)
        print_rank(f"Epoch {epoch}: Average time per sample: {(end_time - start_time) / total_samples:.4f} seconds", rank)

        dist.barrier(device_ids=[torch.cuda.current_device()])
        sampler.set_epoch(epoch)
    
    cleanup()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    world_size = torch.cuda.device_count()
    assert world_size >= 2, "Need at least 2 Devices"
    
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
