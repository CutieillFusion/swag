import matplotlib.pyplot as plt
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from actions import convert_int_to_action, action_meanings
from collections import Counter

class VideoDataset(Dataset):
    def __init__(self, data_paths: List[Tuple[str, str]], sequence_length: int = 64, image_dims: Tuple[int, int] = (30, 32)):
        start_time = time.time()  # Start timing

        self.sequence_length = sequence_length
        self.image_dims = image_dims

        self.frames = []
        self.labels = []
        self.sequence_starting_indices = []
                
        print("Started Processing...")
        with ThreadPoolExecutor() as executor:
            all_data = list(executor.map(self._extract_frames, data_paths))

        # Combine results and compute sequence indices
        cumulative_frames = 0
        for frames, labels in all_data:
            self.sequence_starting_indices.extend(
                [i + cumulative_frames for i in range(0, len(frames), self.sequence_length // 4)
                 if i + self.sequence_length <= len(frames)]
            )
            self.frames.extend(frames)
            self.labels.extend(labels)
            cumulative_frames += len(frames)

        self.labels = np.array(self.labels)

        random.shuffle(self.sequence_starting_indices)
        # Print total time taken to load all video data
        elapsed_time = time.time() - start_time
        print("Finished Processing...")
        print(f"Loaded {len(data_paths)} videos with {len(self.sequence_starting_indices)} total sequences.")
        print(f"Time taken to load all video data: {elapsed_time:.2f} seconds")

    def _extract_frames(self, data_path: Tuple[str, str]):
        video_path = data_path[0]
        label_path = data_path[1]

        print(f"Started processing {video_path} and {label_path}")
        raw_labels = [int(str(label[0]), base=2) for label in pd.read_csv(label_path, header=None).to_numpy()]
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip_interval = max(1, int(original_fps / 30))
        frames = []
        labels = [] 
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip_interval == 0:
                frames.append(frame)
                labels.append(convert_int_to_action(raw_labels[frame_idx]))
            frame_idx += 1

        cap.release()
        print(f"Finished processing {video_path} and {label_path}")
        return frames, labels

    def __len__(self):
        # The number of possible sequences
        return len(self.sequence_starting_indices)
    
    def __getitem__(self, idx):
        sequence = self.frames[idx:idx + self.sequence_length]
        label_sequence = self.labels[idx:idx + self.sequence_length]

        processed_frames = []

        for frame in sequence:
            # Resizes to (256, 240)
            frame_rgb = cv2.resize(frame, self.image_dims)
            # Convert BGR to RGB
            frame_resized = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame_normalized = frame_resized.astype('float32') / 255.0  
            # Convert to Tensor and permute dimensions to (C, H, W)
            tensor_frame = torch.from_numpy(frame_normalized).permute(2, 0, 1)  
            
            processed_frames.append(tensor_frame)

        # Stack all processed frames into a single tensor of shape (sequence_length, C, H, W)
        stacked_sequence = torch.stack(processed_frames)  # Shape: (sequence_length, C, H, W)

        return stacked_sequence, label_sequence


class RandomSequentialSampler(Sampler):
    def __init__(self, data_source: VideoDataset, start_index: int, end_index: int):
        self.data_source = data_source
        self.start_index = start_index
        self.end_index = end_index

    def __iter__(self):
        return iter(self.data_source.sequence_starting_indices[self.start_index:self.end_index])

    def __len__(self):
        return len(self.data_source)

data_paths = [
        ('idm/data/raw/video_4814601.mp4', 'idm/data/raw/labels_4814601.txt'),
        ('idm/data/raw/video_4814603.mp4', 'idm/data/raw/labels_4814603.txt'),
        ('idm/data/raw/video_4814605.mp4', 'idm/data/raw/labels_4814605.txt'),
        ('idm/data/raw/video_4814607.mp4', 'idm/data/raw/labels_4814607.txt'),
        ('idm/data/raw/video_4814609.mp4', 'idm/data/raw/labels_4814609.txt'),
        ('idm/data/raw/video_4814611.mp4', 'idm/data/raw/labels_4814611.txt'),
        ('idm/data/raw/video_4814613.mp4', 'idm/data/raw/labels_4814613.txt'),
        ('idm/data/raw/video_4814615.mp4', 'idm/data/raw/labels_4814615.txt'),
        ('idm/data/raw/video_4814617.mp4', 'idm/data/raw/labels_4814617.txt'),
        ('idm/data/raw/video_4814619.mp4', 'idm/data/raw/labels_4814619.txt'),
        ('idm/data/raw/video_4814621.mp4', 'idm/data/raw/labels_4814621.txt'),
        ('idm/data/raw/video_4814623.mp4', 'idm/data/raw/labels_4814623.txt'),
        ('idm/data/raw/video_4814625.mp4', 'idm/data/raw/labels_4814625.txt'),
        ('idm/data/raw/video_4814627.mp4', 'idm/data/raw/labels_4814627.txt'),
        ('idm/data/raw/video_4814629.mp4', 'idm/data/raw/labels_4814629.txt'),
        ('idm/data/raw/video_4814631.mp4', 'idm/data/raw/labels_4814631.txt'),
    ]

def save_class_histogram(all_labels: torch.Tensor, filename="class_distribution_histogram.png", title="Histogram of Instances per Class"):
    # Convert all_labels to numpy
    all_labels = all_labels.cpu().numpy()

    # Count occurrences of each label
    counter = Counter(all_labels)
    
    # Get unique classes and their counts
    unique_classes = sorted(counter.keys())
    counts = [counter[cls] for cls in unique_classes]
    
    # Map numeric classes to text labels using action_meanings if available
    text_labels = [action_meanings.get(cls, str(cls)) for cls in unique_classes]
    
    # Create figure with increased width
    plt.figure(figsize=(12, 6))
    bars = plt.bar(text_labels, counts, color='skyblue', width=0.5)
    
    # Set the y-axis to a logarithmic scale
    plt.yscale('log')
    plt.ylim(1, max(counts) * 2)  # Set appropriate y-limit
    
    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        xpos = bar.get_x() + bar.get_width() / 2
        # For log scale, offset the text by a fraction of the bar height
        plt.text(xpos, yval * 1.05, int(yval), ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Action')
    plt.ylabel('Number of Instances (Log Scale)')
    plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot as an image file
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    dataset = VideoDataset(data_paths, sequence_length=64)
    sampler = RandomSequentialSampler(dataset, start_index=0, end_index=len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    save_class_histogram(torch.Tensor(dataset.labels), 
                         title="Histogram of Actions in Video Dataset (Log Scale)")

    # for video, labels in dataloader:
    #     # Each batch is a tensor of shape [8, 64, 3, 240, 256]
    #     print(video.shape)
    #     # Each batch is a tensor of shape [8, 64]
    #     print(labels.shape)
    #     # Save each frame as an image
    #     for j, frames in enumerate(video):
    #         for i, frame in enumerate(frames):
    #             if i % 16 != 0:
    #                 continue
    #             # Permute to [H, W, C] format (for PIL compatibility)
    #             frame = frame.permute(1, 2, 0)
                
    #             # Convert to NumPy and scale to [0, 255]
    #             frame = (frame.numpy() * 255).astype('uint8')
                
    #             # Convert to PIL Image and save
    #             img = Image.fromarray(frame)
    #             img.save(f'IDM_VPT/temp/frame_{j}_{i}_{labels[j][i]}.png')
        
    #     print(f"Saved {frames.size(0)} frames as images.")
    #     break