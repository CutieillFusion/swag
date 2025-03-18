import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from actions import convert_int_to_action, action_meanings


def load_labels_from_file(file_path: str) -> List[int]:
    """
    Load labels from a text file.

    Args:
        file_path: Path to the labels file

    Returns:
        List of integer labels
    """
    try:
        with open(file_path, "r") as f:
            labels = f.read().split()
        # Convert binary strings to integers
        labels = [convert_int_to_action(int(label, 2)) for label in labels]
        return labels
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []


def collect_all_labels(directory_path: str) -> List[int]:
    """
    Collect all labels from text files in the given directory and its subdirectories.

    Args:
        directory_path: Path to the directory containing label files

    Returns:
        List of all labels found
    """
    all_labels = []

    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file == "labels.txt":
                    file_path = os.path.join(root, file)
                    labels = load_labels_from_file(file_path)
                    all_labels.extend(labels)
    except Exception as e:
        print(f"Error walking directory {directory_path}: {e}")

    return all_labels


def create_histogram(
    labels: List[int], title: str = "Label Distribution", save_path: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a histogram of the labels with logarithmic scale.

    Args:
        labels: List of integer labels
        title: Title for the histogram
        save_path: Path to save the histogram image (optional)

    Returns:
        Figure and axes objects
    """
    # Count occurrences of each label
    counter = Counter(labels)

    # Get unique classes and their counts
    unique_classes = sorted(counter.keys())
    counts = [counter[cls] for cls in unique_classes]

    # Map numeric classes to text labels using action_meanings
    text_labels = [action_meanings.get(cls, str(cls)) for cls in unique_classes]

    # Increase figure width and reduce bar width for more space between columns
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(text_labels, counts, color="skyblue", width=0.5)

    # Set the y-axis to a logarithmic scale
    ax.set_yscale("log")
    ax.set_ylim(1, 1000000)

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        xpos = bar.get_x() + bar.get_width() / 2
        # For log scale, offset the text by a fraction of the bar height
        ax.text(xpos, yval * 1.05, int(yval), ha="center", va="bottom", fontsize=10)

    # Add labels and title
    ax.set_xlabel("Action")
    ax.set_ylabel("Number of Instances (Unique Frames)")
    ax.set_title(title)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Add grid for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()

    return fig, ax


if __name__ == "__main__":
    # Directory containing the label files
    data_dir = "idm/data/numpy"

    # Collect all labels
    print(f"Collecting labels from {data_dir}...")
    all_labels = collect_all_labels(data_dir)

    print(f"Found {len(all_labels)} labels in total")
    print(f"Unique labels: {sorted(set(all_labels))}")

    # Create and show histogram
    fig, ax = create_histogram(
        all_labels,
        title="Histogram of Unique Actions in Non-overlapping Frames (Log Scale)",
        save_path="label_histogram.png",
    )

    print(f"Histogram saved to label_histogram.png")
    plt.show()
