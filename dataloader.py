import __future__
import os
import re
import random
from typing import Optional, Tuple
import torch
import numpy as np
import torch.nn.functional as F
import time
from actions import convert_int_to_action


class LFUCache:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.cache = {}
        self.usage_count = {}

    def get(self, key: str) -> torch.Tensor | None:
        if key not in self.cache:
            return None
        self.usage_count[key] += 1
        return self.cache[key]

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self.cache:
            self.cache[key] = value
            self.usage_count[key] += 1
        else:
            self.cache[key] = value
            self.usage_count[key] = 1

        if len(self.cache) > self.capacity:
            least_used = min(self.usage_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used]
            del self.usage_count[least_used]

class LRUCache:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.cache = {}
        self.usage_order = []

    def get(self, key: str) -> torch.Tensor | None:
        if key not in self.cache:
            return None
        self.usage_order.remove(key)
        self.usage_order.append(key)
        return self.cache[key]

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self.cache:
            self.cache[key] = value
            self.usage_order.remove(key)
            self.usage_order.append(key)
        else:
            self.cache[key] = value
            self.usage_order.append(key)

        if len(self.cache) > self.capacity:
            lru_key = self.usage_order.pop(0)
            del self.cache[lru_key]


class ChunkedNumpyDataset:
    def __init__(
        self,
        video_ids: list[str],
        data_dir: str,
        sequence_length: int = 64,
        image_dims: tuple[int, int] = (60, 64),
        batch_size: int = 8,
        noop_label: int = 27,
        noop_threshold: int = 32,
        cache_capacity: int = 100,
        cache_type: str = "lfu",
        is_vpt: bool = False,
        has_labels: bool = True,
        stride: int = 16,
        data_splits: dict[str, float] = {"train": 0.8, "val": 0.2},
    ):
        self.video_ids = video_ids
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.image_dims = image_dims
        self.batch_size = batch_size
        self.noop_label = noop_label
        self.noop_threshold = noop_threshold
        self.stride = stride
        self.is_vpt = is_vpt
        self.has_labels = has_labels
        self.movement_classes = [
            67,
            131,
            99,
            163,
            83,
            147,
            97,
            161,
            65,
            81,
            129,
            145,
            98,
            162,
            66,
            82,
            130,
            146,
            96,
            160,
            64,
            80,
            128,
            144,
        ]
        self.movement_classes = [
            convert_int_to_action(movement_class)
            for movement_class in self.movement_classes
        ]
        self.data_splits = data_splits
        assert sum(data_splits.values()) == 1, "Data splits must sum to 1"
        start_time = time.time()

        self.cache_capacity = cache_capacity
        self.cache_type = cache_type

        self.video_info = []
        self.sequence_indices = []
        self.batches = {}

        self._gather_video_metadata()
        print(f"Total videos processed: {len(self.video_info)}")
        self._build_indices()
        print(f"Total sequences: {len(self.sequence_indices)}")
        if self.has_labels:
            self._filter_noops()
            print(f"Total sequences after noop removal: {len(self.sequence_indices)}")
            # self._filter_no_movement()
            # print(
            #     f"Total sequences after no movement removal: {len(self.sequence_indices)}"
            # )
            random.shuffle(self.sequence_indices)

        self._build_batches()

        self._build_cache()

        end_time = time.time()
        print(f"Total time for Video Loading is {end_time - start_time}")

    def set_mode(self, mode: str) -> "ChunkedNumpyDataset":
        self.mode = mode
        return self

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
                print(f"No chunk files found for video {vid_id}. Skipping.")
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
                    print(f"Error reading label file {label_path}: {e}")
                    continue
            total_frames = chunk_files[-1][1]
            if self.has_labels and len(labels) != total_frames:
                print(
                    f"Warning: For video {vid_id}, total_frames={total_frames}, but label count={len(labels)}"
                )
            self.video_info.append(
                {
                    "video_id": vid_id,
                    "chunks": chunk_files,
                    "labels": labels,
                    "num_frames": min(
                        total_frames, len(labels) if self.has_labels else total_frames
                    ),
                }
            )

    def _build_indices(self) -> None:
        for vid_idx, info in enumerate(self.video_info):
            n_frames = info["num_frames"]
            for start in range(0, n_frames - self.sequence_length, self.stride):
                self.sequence_indices.append(
                    (
                        vid_idx,
                        start,
                    )
                )
            
            if not self.has_labels:
                self.sequence_indices.append(
                    (
                        vid_idx,
                        n_frames - self.sequence_length,
                    )
                )

    def _filter_noops(self) -> None:
        if not self.has_labels:
            return
        filtered = []
        for vid_idx, start in self.sequence_indices:
            info = self.video_info[vid_idx]
            seq_labels = info["labels"][start : start + self.sequence_length]
            if not self._has_consecutive_noops(
                seq_labels, self.noop_label, self.noop_threshold
            ):
                filtered.append((vid_idx, start))
        self.sequence_indices = filtered

    def _filter_no_movement(self) -> None:
        if not self.has_labels:
            return
        filtered = []
        for vid_idx, start in self.sequence_indices:
            info = self.video_info[vid_idx]
            seq_labels = info["labels"][start : start + self.sequence_length]
            if self._has_movement(seq_labels, self.movement_classes):
                filtered.append((vid_idx, start))
        self.sequence_indices = filtered

    @staticmethod
    def _has_consecutive_noops(seq, label_val, threshold):
        count = 0
        for label in seq:
            if label == label_val:
                count += 1
                if count >= threshold:
                    return True
            else:
                count = 0
        return False

    @staticmethod
    def _has_movement(seq, movement_classes, threshold=1):
        count = 0
        for label in seq.tolist():
            if label in movement_classes:
                count += 1
                if count >= threshold:
                    return True
        return False

    def __len__(self) -> int:
        return len(self.batches[self.mode])

    def _getitem(self, idx: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        vid_idx, start_frame = self.sequence_indices[idx]
        info = self.video_info[vid_idx]
        end_frame = start_frame + self.sequence_length
        
        label_tensor = None
        if self.has_labels:
            clip_labels = info["labels"][
                (start_frame + 1) if self.is_vpt else start_frame : (
                    (end_frame + 1) if self.is_vpt else end_frame
                )
            ]
            label_tensor = torch.tensor(clip_labels, dtype=torch.long)

        clip_frames = self._load_clip_frames(info["chunks"], start_frame, end_frame)

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

            chunk_tensor = self.chunk_cache.get(chunk_path)
            if chunk_tensor is None:
                chunk_array = np.load(chunk_path, mmap_mode="r+")
                chunk_tensor = torch.from_numpy(chunk_array).float()
                chunk_tensor = F.interpolate(
                    chunk_tensor,
                    size=(self.image_dims[0], self.image_dims[1]),
                    mode="area",
                )
                chunk_tensor = chunk_tensor / 255.0
                self.chunk_cache.put(chunk_path, chunk_tensor)

            offset_in_chunk = current_frame - chunk_start
            chunk_frames_avail = chunk_end - current_frame
            take = min(chunk_frames_avail, frames_needed)
            portion = chunk_tensor[offset_in_chunk : offset_in_chunk + take]
            frames_collected.append(portion)
            frames_needed -= take
            current_frame += take
            if current_frame >= chunk_end:
                chunk_idx += 1
        
        all_frames = torch.cat(frames_collected, dim=0)
        return all_frames

    def _build_batches(self) -> None:
        i = 0
        batches = []
        for _ in range(0, len(self.sequence_indices), self.batch_size):
            batch = []
            for _ in range(self.batch_size):
                batch.append(i)
                i += 1
            batches.append(batch)

        batch_count = len(batches)
        for split_name, split_ratio in self.data_splits.items():
            split_size = int(batch_count * split_ratio)
            split_batches = batches[:split_size]
            batches = batches[split_size:]
            self.batches[split_name] = split_batches

    def _build_cache(self) -> None:
        used_chunks = set()
        for vid_idx, start in self.sequence_indices:
            info = self.video_info[vid_idx]
            sequence_end = start + self.sequence_length
            for chunk_start, chunk_end, chunk_path in info["chunks"]:
                if chunk_end > start and chunk_start < sequence_end:
                    used_chunks.add(chunk_path)
        print(f"Number of unique chunks used: {len(used_chunks)}")

        if self.cache_type == "lru":
            self.chunk_cache = LRUCache(
                capacity=min(self.cache_capacity + 1, len(used_chunks) + 1)
            )
        elif self.cache_type == "lfu":
            self.chunk_cache = LFUCache(
                capacity=min(self.cache_capacity + 1, len(used_chunks) + 1)
            )
        else:
            assert False, f"Invalid cache type: {self.cache_type}"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | list[Tuple[str, int]]]:
        batch = self.batches[self.mode][idx]
        videos, labels = zip(
            *[self._getitem(sequence_index) for sequence_index in batch]
        )
        videos = torch.stack(videos)
        if self.has_labels:
            labels = torch.stack(labels)
        else:
            video_ids = [self.video_info[self.sequence_indices[sequence_index][0]]['video_id'] for sequence_index in batch]
            start_frames = [self.sequence_indices[sequence_index][1] for sequence_index in batch]
            labels = video_ids, start_frames
        return videos, labels


def get_all_videos(directory_path: str, has_labels: bool = True) -> tuple[list[str], int]:
    dirs = []
    file_count = 0
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    if not has_labels or os.path.exists(os.path.join(entry, f"labels.txt")):
                        count = 0
                        with os.scandir(entry) as entries2:
                            for entry2 in entries2:
                                count += 1

                        dirs.append(entry.name)
                        file_count += count - 1
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
    except NotADirectoryError:
        print(f"Error: '{directory_path}' is not a directory.")

    random.shuffle(dirs)

    return dirs, file_count


if __name__ == "__main__":
    dir_path = "vpt/data/numpy"
    ids, cache_capacity = get_all_videos(dir_path, has_labels=False)
    print(f"Upper Limit for Cache capacity {cache_capacity}")

    print(ids)

    video_dataset = ChunkedNumpyDataset(
        video_ids=ids,
        data_dir=dir_path,
        sequence_length=64,
        image_dims=(240, 256),
        batch_size=8,
        has_labels=False,
        is_vpt=False,
        cache_capacity=cache_capacity,
        data_splits={"train": 0.999, "val": 0.001},
        stride=16,
    )

    start_time = time.time()
    for i, (video, label) in enumerate(video_dataset.set_mode("val")):
        pass
    end_time = time.time()
    print(
        f"Total time for {len(video_dataset) * video_dataset.batch_size} Loads is {end_time - start_time}"
    )
    print(
        f"Average time for loading is {(end_time - start_time) / (len(video_dataset) * video_dataset.batch_size)}"
    )

    start_time = time.time()
    for i, (video, label) in enumerate(video_dataset.set_mode("val")):
        pass
    end_time = time.time()
    print(
        f"Total time for {len(video_dataset) * video_dataset.batch_size} Loads is {end_time - start_time}"
    )
    print(
        f"Average time for loading is {(end_time - start_time) / (len(video_dataset) * video_dataset.batch_size)}"
    )
