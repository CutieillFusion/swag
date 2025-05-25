import cv2
import numpy as np
import argparse
import os
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert a video to numpy sequences")
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="Video id used to construct the video filepath",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="idm/data/raw",
        help="Directory for input videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="idm/data/numpy",
        help="Directory for output numpy sequences",
    )
    parser.add_argument(
        "--width", type=int, default=256, help="Width of the output video"
    )
    parser.add_argument(
        "--height", type=int, default=240, help="Height of the output video"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=8,
        help="Number of continuous frames per sequence",
    )
    parser.add_argument(
        "--chunk_length", type=int, default=1, help="Number of sequences per chunk"
    )
    parser.add_argument(
        "--labels",
        type=bool,
        default=True,
        help="Whether to save labels",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    frames = []

    if (
        os.path.exists(os.path.join(args.output_dir, f"{args.video_id}/labels.txt"))
        and not args.labels
    ):
        print(f"Video id {args.video_id} already Finished")
        return

    video_filepath = os.path.join(args.input_dir, f"video_{args.video_id}.mp4")
    if not os.path.exists(video_filepath):
        print(f"Video file {video_filepath} does not exist.")
        return

    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")
    frame_skip_interval = max(1, int(original_fps / 30))
    print(f"Frame skip interval: {frame_skip_interval}")

    frame_idx = 0
    saved_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip_interval == 0:
            frame_resized = cv2.resize(frame, (args.width, args.height))
            frames.append(frame_resized)
            saved_frames += 1
        frame_idx += 1

    cap.release()

    if not frames:
        print("No frames were read from the video.")
        return

    video = np.stack(frames, axis=0)
    video_tchw = video.transpose(0, 3, 1, 2)
    print(video_tchw.shape)
    total_frames = video_tchw.shape[0]
    seq_len = args.sequence_length
    chunk_length = args.chunk_length
    num_sequences = total_frames // (seq_len * chunk_length) + 1
    if num_sequences == 0:
        print(f"Not enough frames ({total_frames}) for a sequence of length {seq_len}.")
        return

    print(f"Video id {args.video_id} of shape {video_tchw.shape}")

    os.makedirs(os.path.join(args.output_dir, f"{args.video_id}"), exist_ok=True)

    for i in range(num_sequences):
        start = i * seq_len * chunk_length
        end = min(start + seq_len * chunk_length, total_frames)
        sequence = video_tchw[start:end]
        output_path = os.path.join(
            args.output_dir, f"{args.video_id}/frames_{start}_{end}.npy"
        )
        np.save(output_path, sequence)

    if args.labels:
        with open(
            os.path.join(args.input_dir, f"labels_{args.video_id}.txt"), "r"
        ) as f:
            labels = f.readlines()

        sampled_labels = labels[::frame_skip_interval]

        with open(
            os.path.join(args.output_dir, f"{args.video_id}/labels.txt"), "w"
        ) as f:
            f.writelines(sampled_labels)

    elapsed_time = time.time() - start_time

    print(f"Saved {saved_frames} frames to numpy sequences")
    print(f"Processing completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
