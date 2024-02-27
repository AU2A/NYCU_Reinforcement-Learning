from racecar_gym.env_eval import RaceEnv
import cv2
from collections import deque
from stable_baselines3 import PPO
import numpy as np

env = RaceEnv(
    scenario="austria_competition_collisionStop",  # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
    render_mode="rgb_array_birds_eye",
    reset_when_collision=False,  # Only work for 'austria_competition' and 'austria_competition_collisionStop'
)


def observation_postprocess(self, obs):
    obs = obs[self.camera_name]
    obs = cv2.resize(obs, (84, 84))
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    return obs


model = PPO.load("log/PPO_6/best_model/best_model.zip")

videoframe = []

cnt = 0

frames = deque(maxlen=8)
obs, info = env.reset()
observation = cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
observation = cv2.resize(observation, (84, 84))
for i in range(7):
    frames.append(observation)

terminated = False
while not terminated and cnt < 2000:
    cnt += 1
    videoframe.append(env.render())

    observation = cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (84, 84))

    frames.append(observation)

    obs = np.stack(frames, axis=0)

    action = model.predict(obs, deterministic=True)
    action[0][0] = action[0][0] * 0.8
    print(cnt, action[0], info["progress"])
    obs, rew, terminated, truncated, info = env.step(action[0])


height, width, layers = videoframe[0].shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_path = "temp/video.mp4"
video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
for frame in videoframe:
    video.write(frame)
cv2.destroyAllWindows()
video.release()
