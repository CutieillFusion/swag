from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
from actions import ACTION_SPACE
import torch
from wrappers import apply_wrappers
from vpt.vpt import VPT
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the VPT model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()
    return args

args = parse_arguments()

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym_super_mario_bros.make(
    "SuperMarioBrosRandomStages-v0", render_mode="rgb_array", apply_api_compatibility=True
)
env = JoypadSpace(env, ACTION_SPACE)
env = apply_wrappers(env, args.output_dir)

print("Observation Space:", env.observation_space.shape)
print("Action Space:", env.action_space)

vpt = VPT(
    n_actions=len(ACTION_SPACE),
    input_dim=(64, 3, 64, 60),
    feature_channels=[32, 64, 64],
    transformer_blocks=2,
    transformer_heads=4,
    ff_dim=512,
    embedding_dim=512,
    freeze=True,
)

def append_values(observations, rewards, dones, observation, reward, done):
    observations.append(observation)
    rewards.append(reward)
    dones.append(done)

def process_observation(observation):
    observation = torch.from_numpy(np.asarray(observation)).float() / 255.0
    observation = observation.permute(2, 0, 1)
    return observation.to(device)

def get_filtered_logits(logits, threshold: float = 0.9):
    sorted_probs, indices = torch.sort(logits[0][-1], descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff = torch.where(cumsum_probs > threshold)[0][0] + 1
    filtered_probs = torch.zeros_like(logits[0][-1])
    filtered_probs[indices[:cutoff]] = logits[0][-1][indices[:cutoff]]
    filtered_probs = filtered_probs / filtered_probs.sum()
    action_index = torch.multinomial(filtered_probs, 1).item()
    return filtered_probs, action_index

vpt.load_model(args.model_path)
vpt = vpt.to(device)
vpt.eval()
env.reset()

observations, rewards, dones = [], [], []

observation, reward, done, trunc, info = env.step(action=0)
append_values(observations, rewards, dones, observation, reward, done)

epochs = 0

while True:

    if len(observations) > 64:
        logits = vpt(process_observation(observation))
        _, action_index = get_filtered_logits(logits)

    observation, reward, done, truncated, info = env.step(action_index)
    append_values(observations, rewards, dones, observation, reward, done)


    if done:
        observation, info = env.reset()
        print("reset")
        epochs += 1
        if epochs > 64:
            break

env.close()
