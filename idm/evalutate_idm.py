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


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float, float]:
    probs = F.softmax(logits, dim=1)
    _, predicted = torch.max(probs, dim=1)

    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    precision = precision_score(labels, predicted, average="macro", zero_division=0)
    recall = recall_score(labels, predicted, average="macro", zero_division=0)
    f1 = f1_score(labels, predicted, average="macro", zero_division=0)

    return precision, recall, f1


def save_confusion_matrix(
    all_labels: torch.Tensor, all_logits: torch.Tensor, output_dir: str
) -> None:
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


def save_class_histogram(
    all_labels: torch.Tensor, all_logits: torch.Tensor, output_dir: str
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
    plt.savefig(os.path.join(output_dir, "class_histogram.png"))
    plt.close()


def save_prediction_class_accuracy_histogram(
    all_labels: torch.Tensor, all_logits: torch.Tensor, output_dir: str
) -> None:
    probs = F.softmax(all_logits, dim=1)
    _, predicted = torch.max(probs, dim=1)

    all_labels = all_labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    correct_predictions = predicted == all_labels

    unique_classes = np.unique(all_labels)
    class_counts = {cls: np.sum(all_labels == cls) for cls in unique_classes}
    correct_counts = {
        cls: np.sum(correct_predictions[all_labels == cls]) for cls in unique_classes
    }
    accuracy_percentages = {
        cls: (
            (correct_counts[cls] / class_counts[cls])
            if class_counts[cls] > 0
            else float("nan")
        )
        for cls in unique_classes
    }

    sorted_classes = sorted(correct_counts.keys())
    percentages = [accuracy_percentages[cls] for cls in sorted_classes]
    text_labels = [action_meanings[cls] for cls in sorted_classes]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(text_labels, percentages, color="skyblue", width=0.6)

    for bar, cls in zip(bars, sorted_classes):
        height = bar.get_height()
        if np.isnan(height):
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.5,
                "N/A",
                ha="center",
                va="bottom",
            )
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Class")
    plt.xticks(rotation=90)
    plt.ylim(0, 1.05)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_class_accuracy_histogram.png"))
    plt.close()


def evaluate_model(
    model: IDM, dataset: ChunkedNumpyDataset, device: torch.device, output_dir: str
) -> None:
    model.eval()

    test_correct_predictions_top1 = 0
    test_correct_predictions_top3 = 0
    test_total_predictions = 0
    test_all_logits = []
    test_all_labels = []

    with torch.no_grad():
        for videos, labels in dataset.set_mode("val"):
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

    all_preds = torch.argmax(test_all_logits, dim=1).cpu().numpy()
    all_labels_np = test_all_labels.cpu().numpy()
    unique_classes = np.unique(all_labels_np)

    class_precision = []
    class_recall = []
    class_f1 = []
    
    for cls in unique_classes:
        true_positives = np.sum((all_preds == cls) & (all_labels_np == cls))
        false_positives = np.sum((all_preds == cls) & (all_labels_np != cls))
        false_negatives = np.sum((all_preds != cls) & (all_labels_np == cls))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)
    
    macro_precision = np.mean(class_precision)
    macro_recall = np.mean(class_recall)
    macro_f1 = np.mean(class_f1)

    test_top1_accuracy = 100 * test_correct_predictions_top1 / test_total_predictions
    test_top3_accuracy = 100 * test_correct_predictions_top3 / test_total_predictions

    print(f"Test Results:")
    print(f"Top-1 Accuracy: {test_top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {test_top3_accuracy:.2f}%")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Macro-Precision: {macro_precision:.4f}")
    print(f"Macro-Recall: {macro_recall:.4f}")
    print(f"Macro-F1 Score: {macro_f1:.4f}")

    save_confusion_matrix(test_all_labels, test_all_logits, output_dir)
    save_class_histogram(test_all_labels, test_all_logits, output_dir)
    save_prediction_class_accuracy_histogram(
        test_all_labels, test_all_logits, output_dir
    )

    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Test Results:\n")
        f.write(f"Top-1 Accuracy: {test_top1_accuracy:.2f}%\n")
        f.write(f"Top-3 Accuracy: {test_top3_accuracy:.2f}%\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"Macro-Precision: {macro_precision:.4f}\n")
        f.write(f"Macro-Recall: {macro_recall:.4f}\n")
        f.write(f"Macro-F1 Score: {macro_f1:.4f}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IDM model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="idm/results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument("--stride", type=int, default=32, help="Stride for evaluation")
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
        default=512,
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

    dir_path = "idm/data/numpy"
    ids, cache_capacity = get_all_videos(dir_path)

    dataset = ChunkedNumpyDataset(
        video_ids=ids,
        data_dir=dir_path,
        sequence_length=input_dims[0],
        image_dims=(input_dims[2], input_dims[3]),
        batch_size=args.batch_size,
        cache_capacity=cache_capacity,
        is_vpt=False,
        stride=args.stride,
        data_splits={"val": 1.0},
    )

    evaluate_model(model, dataset, device, args.output_dir)

    print(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
