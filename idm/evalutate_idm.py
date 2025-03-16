import argparse
import torch
import os
import numpy as np
from torch.nn import functional as F
from idm import IDM
from dataloader import ChunkedNumpyDataset, get_all_videos
from actions import ACTION_SPACE, action_meanings
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    _, pred = torch.topk(logits, k=k, dim=1)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.sum().item()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    probs = F.softmax(logits, dim=1)
    _, predicted = torch.max(probs, dim=1)

    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    precision = precision_score(labels, predicted, average="macro", zero_division=0)
    recall = recall_score(labels, predicted, average="macro", zero_division=0)
    f1 = f1_score(labels, predicted, average="macro", zero_division=0)

    return precision, recall, f1


def save_confusion_matrix(all_labels: torch.Tensor, all_logits: torch.Tensor, output_dir: str) -> None:
    probs = F.softmax(all_logits, dim=1)
    _, predicted = torch.max(probs, dim=1)

    all_labels = all_labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    unique_classes, counts = np.unique(all_labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    top10_classes = unique_classes[sorted_indices][:10]

    top10_labels = [action_meanings[c] for c in top10_classes]

    cm = confusion_matrix(all_labels, predicted, labels=top10_classes)
    cm_log = np.log1p(cm)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm_log,
        annot=cm,   
        fmt="d",
        cmap="Blues",
        xticklabels=top10_labels,
        yticklabels=top10_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Top 10 Largest Classes) - Log Scale")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def save_prediction_class_histogram(all_labels: torch.Tensor, all_logits: torch.Tensor, output_dir: str) -> None:
    probs = F.softmax(all_logits, dim=1)
    _, predicted = torch.max(probs, dim=1)

    all_labels = all_labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    correct_predictions = predicted == all_labels

    unique_classes = np.unique(all_labels)
    correct_counts = {
        cls: np.sum(correct_predictions[all_labels == cls]) for cls in unique_classes
    }

    sorted_classes = sorted(correct_counts.keys())
    counts = [correct_counts[cls] for cls in sorted_classes]
    text_labels = [action_meanings[cls] for cls in sorted_classes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(text_labels, counts, color="skyblue")
    
    plt.yscale('log')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height*1.05,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.xlabel("Class")
    plt.ylabel("Number of Correct Predictions (Log Scale)")
    plt.title("Histogram of Correct Predictions per Class (Log Scale)")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(os.path.join(output_dir, "prediction_class_histogram.png"))
    plt.close()


def evaluate_model(
    model: IDM,
    test_dataloader: ChunkedNumpyDataset,
    device: torch.device,
    output_dir: str
) -> None:
    model.eval()
    
    test_correct_predictions_top1 = 0
    test_correct_predictions_top3 = 0
    test_total_predictions = 0
    test_all_logits = []
    test_all_labels = []

    with torch.no_grad():
        for videos, labels in test_dataloader:
            videos, labels = videos.to(device), labels.to(device)

            logits = model(videos)

            if logits.size(1) > 32 and labels.size(0) > 32:
                logits = logits[:, 16:-16, :]
                labels = labels[:, 16:-16]

            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)

            if logits.size(0) == 0 or labels.size(0) == 0:
                continue

            _, predicted = torch.max(logits, dim=1)
            test_correct_predictions_top1 += (predicted == labels).sum().item()
            test_correct_predictions_top3 += top_k_accuracy(logits, labels, k=3)
            test_total_predictions += labels.size(0)

            test_all_logits.append(logits)
            test_all_labels.append(labels)

    test_all_logits = torch.cat(test_all_logits, dim=0)
    test_all_labels = torch.cat(test_all_labels, dim=0)

    test_precision, test_recall, test_f1 = compute_metrics(
        test_all_logits, test_all_labels
    )

    test_top1_accuracy = 100 * test_correct_predictions_top1 / test_total_predictions
    test_top3_accuracy = 100 * test_correct_predictions_top3 / test_total_predictions

    print(f"Test Results:")
    print(f"Top-1 Accuracy: {test_top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {test_top3_accuracy:.2f}%")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    # Save visualizations
    save_confusion_matrix(test_all_labels, test_all_logits, output_dir)
    save_prediction_class_histogram(test_all_labels, test_all_logits, output_dir)

    # Save metrics to file
    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Test Results:\n")
        f.write(f"Top-1 Accuracy: {test_top1_accuracy:.2f}%\n")
        f.write(f"Top-3 Accuracy: {test_top3_accuracy:.2f}%\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IDM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="idm", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--stride", type=int, default=16, help="Stride for evaluation")
    parser.add_argument("--x", type=int, default=32, help="Dimension of x")
    parser.add_argument("--y", type=int, default=30, help="Dimension of y")
    parser.add_argument("--feature_channels", type=str, default="32,64,64", help="Comma-separated list of feature channels")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of embeddings")
    parser.add_argument("--ff_dim", type=int, default=256, help="Dimension of feed-forward networks in transformer")
    parser.add_argument("--transformer_heads", type=int, default=4, help="Number of attention heads in transformer")
    parser.add_argument("--transformer_blocks", type=int, default=1, help="Number of transformer blocks")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration
    input_dims = (64, 3, args.y, args.x)
    feature_channels = list(map(int, args.feature_channels.split(",")))
    
    # Initialize model
    model = IDM(
        n_actions=len(ACTION_SPACE),
        input_dim=input_dims,
        feature_channels=feature_channels,
        transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
        ff_dim=args.ff_dim,
        embedding_dim=args.embedding_dim,
    )
    
    # Load model weights
    model.load_model(args.model_path)
    model = model.to(device)
    
    # Prepare test dataset
    dir_path = "idm/data/numpy"
    ids, cache_capacity = get_all_videos(dir_path)
    
    test_dataset = ChunkedNumpyDataset(
        video_ids=ids,
        data_dir=dir_path,
        sequence_length=input_dims[0],
        image_dims=(input_dims[2], input_dims[3]),
        batch_size=args.batch_size,
        cache_capacity=cache_capacity,
        is_vpt=False,
        stride=args.stride,
    )
    
    # Evaluate model
    evaluate_model(model, test_dataset, device, args.output_dir)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
