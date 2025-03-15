import argparse
from sklearn.utils.class_weight import compute_class_weight
import torch
import os
import time
from idm import IDM
import torch
from torch import nn
from torch.nn import functional as F
from dataloader import ChunkedNumpyDataset, get_all_videos
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from actions import ACTION_SPACE, action_meanings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def top_k_accuracy(logits, labels, k=3):
    _, pred = torch.topk(logits, k=k, dim=1)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.sum().item()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    probs = F.softmax(logits, dim=1)

    _, predicted = torch.max(probs, dim=1)

    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    precision = precision_score(labels, predicted, average="macro", zero_division=0)
    recall = recall_score(labels, predicted, average="macro", zero_division=0)
    f1 = f1_score(labels, predicted, average="macro", zero_division=0)

    return precision, recall, f1


def save_confusion_matrix(all_labels, all_logits, filename="confusion_matrix.png"):
    probs = torch.nn.functional.softmax(all_logits, dim=1)
    _, predicted = torch.max(probs, dim=1)

    all_labels = all_labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    unique_classes, counts = np.unique(all_labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    top10_classes = unique_classes[sorted_indices][:10]

    top10_labels = [action_meanings[c] for c in top10_classes]

    cm = confusion_matrix(all_labels, predicted, labels=top10_classes)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=top10_labels,
        yticklabels=top10_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Top 10 Largest Classes)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_prediction_class_histogram(
    all_labels, all_logits, filename="prediction_class_histogram.png"
):
    probs = torch.nn.functional.softmax(all_logits, dim=1)
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
    plt.bar(text_labels, counts, color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Number of Correct Predictions")
    plt.title("Histogram of Correct Predictions per Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compute_class_weights(dataloader, num_classes, device):
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

    # Use mean weight for unseen classes instead of extreme value
    mean_weight = np.mean(list(class_weights_dict.values()))
    class_weights = np.full(num_classes, mean_weight, dtype=np.float32)
    for cls, weight in class_weights_dict.items():
        class_weights[cls] = weight

    return torch.tensor(class_weights, dtype=torch.float).to(device)


def train_idm(
    model: nn.Module,
    train_dataloader: ChunkedNumpyDataset,
    val_dataloader: ChunkedNumpyDataset,
    num_epochs: int,
    starting_epoch: int = 0,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-5,
    patience: int = 15,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    job_id: int = 0,
):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.33, patience=7, min_lr=1e-9
    )

    best_val_loss = float("inf")
    best_val_loss_result = ""
    patience_counter = 0

    current_epoch = 0

    for epoch in range(num_epochs):
        current_epoch = epoch + starting_epoch + 1

        model.train()
        training_loss = 0
        correct_predictions_top1 = 0
        correct_predictions_top3 = 0
        total_predictions = 0
        all_logits = []
        all_labels = []

        for videos, labels in train_dataloader:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(videos)

            if logits.size(1) > 32 and labels.size(0) > 32:
                logits = logits[:, 16:-16, :]
                labels = labels[:, 16:-16]

            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)

            if logits.size(0) == 0 or labels.size(0) == 0:
                print("Skipping batch due to empty logits or labels.")
                continue

            loss = criterion(logits, labels)

            if torch.isnan(loss):
                print("NaN loss detected! Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            training_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct_predictions_top1 += (predicted == labels).sum().item()
            correct_predictions_top3 += top_k_accuracy(logits, labels)
            total_predictions += labels.size(0)

            all_logits.append(logits)
            all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        precision, recall, f1 = compute_metrics(all_logits, all_labels)

        train_top1_accuracy = 100 * correct_predictions_top1 / total_predictions
        train_top3_accuracy = 100 * correct_predictions_top3 / total_predictions
        training_loss = training_loss / len(train_dataloader)

        model.eval()
        val_loss = 0
        val_correct_predictions_top1 = 0
        val_correct_predictions_top3 = 0
        val_total_predictions = 0
        val_all_logits = []
        val_all_labels = []

        with torch.no_grad():
            for videos, labels in val_dataloader:
                videos, labels = videos.to(device), labels.to(device)

                logits = model(videos)

                if logits.size(1) > 32 and labels.size(0) > 32:
                    logits = logits[:, 16:-16, :]
                    labels = labels[:, 16:-16]

                logits = logits.reshape(-1, logits.size(-1))
                labels = labels.reshape(-1)

                if logits.size(0) == 0 or labels.size(0) == 0:
                    continue

                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits, dim=1)
                val_correct_predictions_top1 += (predicted == labels).sum().item()
                val_correct_predictions_top3 += top_k_accuracy(logits, labels)
                val_total_predictions += labels.size(0)

                val_all_logits.append(logits)
                val_all_labels.append(labels)

        val_all_logits = torch.cat(val_all_logits, dim=0)
        val_all_labels = torch.cat(val_all_labels, dim=0)

        val_precision, val_recall, val_f1 = compute_metrics(
            val_all_logits, val_all_labels
        )

        val_top1_accuracy = 100 * val_correct_predictions_top1 / val_total_predictions
        val_top3_accuracy = 100 * val_correct_predictions_top3 / val_total_predictions
        val_loss = val_loss / len(val_dataloader)

        print(
            f"Epoch [{current_epoch}/{num_epochs}] Training Loss: {training_loss:.4f}, Training Top-1 Accuracy: {train_top1_accuracy:.2f}%, Training Top-3 Accuracy: {train_top3_accuracy:.2f}%, Training Precision: {precision:.2f}, Training Recall: {recall:.2f}, Training F1: {f1:.2f}"
        )
        print(
            f"Epoch [{current_epoch}/{num_epochs}] Validation Loss: {val_loss:.4f}, Validation Top-1 Accuracy: {val_top1_accuracy:.2f}%, Validation Top-3 Accuracy: {val_top3_accuracy:.2f}%, Validation Precision: {val_precision:.2f}, Validation Recall: {val_recall:.2f}, Validation F1: {val_f1:.2f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss_result = f"Epoch [{current_epoch}/{num_epochs}] Validation Loss: {val_loss:.4f}, Validation Top-1 Accuracy: {val_top1_accuracy:.2f}%, Validation Top-3 Accuracy: {val_top3_accuracy:.2f}%, Validation Precision: {val_precision:.2f}, Validation Recall: {val_recall:.2f}, Validation F1: {val_f1:.2f}"
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join("idm/models", str(job_id))
            os.makedirs(model_path, exist_ok=True)
            if isinstance(model, torch.nn.DataParallel):
                best_model = model.module
            else:
                best_model = model
            best_model.save_model(f"{model_path}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {current_epoch}")
                break

    save_prediction_class_histogram(
        all_labels, all_logits, filename=f"{model_path}/prediction_class_histogram.png"
    )
    save_confusion_matrix(
        all_labels, all_logits, filename=f"{model_path}/confusion_matrix.png"
    )
    print("Training complete.")
    print(f"[Best Epoch {best_val_loss_result}")
    return current_epoch


def measure_inference_time(
    model: nn.Module, dataloader: ChunkedNumpyDataset, device: torch.device
):
    model.eval()

    total_start_time = time.time()
    num_samples = 0

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            model(videos)
            num_samples += videos.size(0)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    print(f"[All Batches] Total inference time (seconds): {total_time:.6f}")
    print(
        f"[All Batches] Average inference time per sample (seconds): {total_time / num_samples:.6f}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Argument Parser for Model Configuration"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=256, help="Dimension of embeddings"
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=256,
        help="Dimension of feed-forward networks in transformer",
    )
    parser.add_argument(
        "--transformer_heads",
        type=int,
        default=4,
        help="Number of attention heads in transformer",
    )
    parser.add_argument(
        "--transformer_blocks", type=int, default=1, help="Number of transformer blocks"
    )
    parser.add_argument("--x", type=int, default=32, help="Dimension of x")
    parser.add_argument("--y", type=int, default=30, help="Dimension of y")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument("--job_id", type=str, required=True, help="Job identifier")
    parser.add_argument(
        "--feature_channels",
        type=str,
        default="32,64,64",
        help="Comma-separated list of feature channels",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Stride for training"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    model_path = os.path.join("idm/models", str(args.job_id))
    os.makedirs(model_path, exist_ok=True)

    input_dims = (64, 3, args.y, args.x)
    print(f"Input Dims {input_dims}")
    feature_channels = list(map(int, args.feature_channels.split(",")))
    print(f"Feature Channels {feature_channels}")
    transformer_blocks = args.transformer_blocks
    print(f"Transformer Blocks {transformer_blocks}")
    transformer_heads = args.transformer_heads
    print(f"Transformer Heads {transformer_heads}")
    ff_dim = args.ff_dim
    print(f"FeedForward Dimension {ff_dim}")
    embedding_dim = args.embedding_dim
    print(f"Embedding Dimension {embedding_dim}")
    learning_rate = args.learning_rate
    print(f"Learning Rate {learning_rate}")
    weight_decay = args.weight_decay
    print(f"Weight Decay {weight_decay}")

    model = IDM(
        n_actions=len(ACTION_SPACE),
        input_dim=input_dims,
        feature_channels=feature_channels,
        transformer_blocks=transformer_blocks,
        transformer_heads=transformer_heads,
        ff_dim=ff_dim,
        embedding_dim=embedding_dim,
    )
    model.print_model_parameters()

    dir_path = "idm/data/numpy"
    ids, cache_capacity = get_all_videos(dir_path)
    print(f"Upper Limit for Cache capacity {cache_capacity}")
    split_index = int(len(ids) * 0.9375)

    training_dataset = ChunkedNumpyDataset(
        video_ids=ids[:split_index], 
        data_dir=dir_path,
        sequence_length=input_dims[0],
        image_dims=(input_dims[2], input_dims[3]),
        batch_size=args.batch_size,
        cache_capacity=cache_capacity,
        is_vpt=False,
        stride=args.stride,
    )

    validation_dataset = ChunkedNumpyDataset(
        video_ids=ids[split_index:],
        data_dir=dir_path,
        sequence_length=input_dims[0],
        image_dims=(input_dims[2], input_dims[3]),
        batch_size=args.batch_size,
        cache_capacity=cache_capacity,
        is_vpt=False,
        stride=args.stride,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    start_time = time.time()

    num_classes = len(ACTION_SPACE)
    class_weights = compute_class_weights(training_dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_idm(
        model,
        training_dataset,
        validation_dataset,
        num_epochs=1000,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        criterion=criterion,
        job_id=args.job_id,
    )

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")


if __name__ == "__main__":
    main()
