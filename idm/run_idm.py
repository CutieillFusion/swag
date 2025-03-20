import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from idm import IDM
from dataloader import ChunkedNumpyDataset, get_all_videos
from actions import ACTION_SPACE, action_meanings

def save_class_histogram(
    all_labels: torch.Tensor, file_path: str
) -> None:
    all_labels = all_labels.cpu().numpy()

    unique_classes, counts = np.unique(all_labels, return_counts=True)

    sorted_indices = np.argsort(unique_classes)
    sorted_classes = unique_classes[sorted_indices]
    sorted_counts = counts[sorted_indices]

    text_labels = [action_meanings[cls] for cls in sorted_classes]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(text_labels, sorted_counts, color="skyblue", width=0.6)

    plt.yscale("log")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.05,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Class")
    plt.ylabel("Number of Samples (Log Scale)")
    plt.title("Histogram of Class Distribution (Log Scale)")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def run_model(
    model: IDM, dataset: ChunkedNumpyDataset, device: torch.device, output_dir: str
) -> None:
    
    model.eval()

    current_video = None
    current_start_frame = None
    video_labels = np.array([])

    with torch.no_grad():
        for videos, video_info in dataset.set_mode("labels"):
            video_ids = video_info[0]
            start_frames = video_info[1]

            videos = videos.to(device)

            logits = model(videos)

            logits = logits.reshape(-1, logits.size(-1))

            _, predicted = torch.max(logits, dim=1)
            
            if current_video != video_ids[0]:
                if current_video is not None:
                    os.makedirs(f"{output_dir}/{current_video}", exist_ok=True)
                    output_file = os.path.join(f"{output_dir}/{current_video}", "labels.txt")

                    with open(output_file, "w") as f:
                        for label in video_labels:
                            binary_label = bin(label)[2:].zfill(8)
                            f.write(f"{binary_label}\n")

                    print(f"Saved labels for video {current_video} to {output_file}")
                    
                    histogram_file = os.path.join(f"{output_dir}/{current_video}", "class_distribution_histogram.png")
                    save_class_histogram(torch.tensor(video_labels), histogram_file)

                current_video = video_ids[0]
                video_labels = predicted.cpu()[:-16].numpy()
                print("Started labelling video:", current_video)
            else:
                removed_frames = 16
                if (start_frames[0] - current_start_frame) < 32:
                    removed_frames = start_frames[0] - current_start_frame
                video_labels = np.concatenate((video_labels[:-(32 - removed_frames)], predicted.cpu()[16:].numpy()))
            current_start_frame = start_frames[0]

    if current_video is not None:
        os.makedirs(f"{output_dir}/{current_video}", exist_ok=True)
        output_file = os.path.join(f"{output_dir}/{current_video}", "labels.txt")

        with open(output_file, "w") as f:
            for label in video_labels:
                binary_label = bin(label)[2:].zfill(8)
                f.write(f"{binary_label}\n")

        print(f"Saved labels for video {current_video} to {output_file}")
        
        histogram_file = os.path.join(f"{output_dir}/{current_video}", "class_distribution_histogram.png")
        save_class_histogram(torch.tensor(video_labels), histogram_file)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label with IDM model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="vpt/data/numpy",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vpt/data/numpy",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument("--x", type=int, default=64, help="Dimension of x")
    parser.add_argument("--y", type=int, default=60, help="Dimension of y")
    parser.add_argument(
        "--feature_channels",
        type=str,
        default="32,64,64",
        help="Comma-separated list of feature channels",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=512, help="Dimension of embeddings"
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=2048,
        help="Dimension of feed-forward networks in transformer",
    )
    parser.add_argument(
        "--transformer_heads",
        type=int,
        default=4,
        help="Number of attention heads in transformer",
    )
    parser.add_argument(
        "--transformer_blocks", type=int, default=2, help="Number of transformer blocks"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dims = (64, 3, args.y, args.x)
    feature_channels = list(map(int, args.feature_channels.split(",")))

    model = IDM(
        n_actions=len(ACTION_SPACE),
        input_dim=input_dims,
        feature_channels=feature_channels,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
        ff_dim=args.ff_dim,
        embedding_dim=args.embedding_dim,
    )

    model.load_model(args.model_path)
    model = model.to(device)

    dir_path = args.input_dir
    ids, _ = get_all_videos(dir_path, has_labels=False)

    labeled_ids, _ = get_all_videos(dir_path)
    ids = [id for id in ids if id not in labeled_ids]

    dataset = ChunkedNumpyDataset(
        video_ids=ids,
        data_dir=dir_path,
        sequence_length=64,
        image_dims=(input_dims[2], input_dims[3]),
        batch_size=1,
        has_labels=False,
        is_vpt=False,
        cache_capacity=10,
        cache_type="lfu",
        data_splits={"labels": 1.0},
        stride=32,
    )

    run_model(model, dataset, device, args.output_dir)

    print(f"Labeling with IDM complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
