import os
from gym import Wrapper
from gym.wrappers import ResizeObservation, FrameStack
import cv2
import numpy as np
from actions import action_meanings

class RecordVideo(Wrapper):
    def __init__(self, env, video_folder):
        super().__init__(env)
        self.video_folder = video_folder
        self.frame_count = 0
        self.episode_id = 0

    def step(self, action):
        observation, reward, done, truncate, info = self.env.step(action)
        os.makedirs(os.path.join(self.video_folder, f"episode_{self.episode_id}"), exist_ok=True)
        
        h, w = observation.shape[:2]
        text = action_meanings[action]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = w - text_size[0] - 10
        text_y = h - 10
        
        cv2.putText(observation, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        cv2.imwrite(os.path.join(self.video_folder, f"episode_{self.episode_id}/frame_{self.frame_count}.png"), observation)
        self.frame_count += 1
        return observation, reward, done, truncate, info

    def reset(self, **kwargs):
        if self.frame_count > 0:
            episode_dir = os.path.join(self.video_folder, f"episode_{self.episode_id}")
            frames = []
            for i in range(self.frame_count):
                frame_path = os.path.join(episode_dir, f"frame_{i}.png")
                frame = cv2.imread(frame_path)
                frames.append(frame)
                os.remove(frame_path)
            
            height, width = frames[0].shape[:2]
            video_path = os.path.join(episode_dir, "episode.mp4")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
        self.frame_count = 0
        self.episode_id += 1
        return self.env.reset(**kwargs)

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, truncate, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, reward, done, truncate, info

class RGBConverter(Wrapper):
    def step(self, action):
        observation, reward, done, truncate, info = self.env.step(action)
        observation = cv2.cvtColor(np.array(observation), cv2.COLOR_BGR2RGB)
        return observation, reward, done, truncate, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = cv2.cvtColor(np.array(observation), cv2.COLOR_BGR2RGB)
        return observation, info

def apply_wrappers(env, video_folder: str = "emulator/videos"):
    env = RGBConverter(env)
    env = RecordVideo(env, video_folder=video_folder)
    env = ResizeObservation(env, shape=(64, 60))
    env = SkipFrame(env, skip=10)
    return env
