import gym
import cv2
import gymnasium as gym
import numpy as np
from numpy import array, float32

from racecar_gym.env_ppo import RaceEnv
from collections import OrderedDict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env():
    env = RaceEnv(
        scenario="austria_competition_collisionStop",  # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
        render_mode="rgb_array_birds_eye",
        reset_when_collision=False,  # Only work for 'austria_competition' and 'austria_competition_collisionStop'
    )
    return env


dir_name = "PPO_9"

checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path="./log/" + dir_name + "/", name_prefix="PPO"
)

eval_callback = EvalCallback(
    make_env(),
    best_model_save_path="./log/" + dir_name + "/best_model",
    log_path="./log/" + dir_name + "/results",
    eval_freq=10000,
)
if __name__ == "__main__":
    env = SubprocVecEnv([lambda: make_env() for i in range(1)])
    # env = DummyVecEnv([lambda: make_env()])

    model = PPO("CnnPolicy", env=env, verbose=1, tensorboard_log="./log/")
    # PPO1
    # model = PPO.load(
    #     "log/Hank/best_model_10240.zip", env=env, verbose=1, tensorboard_log="./log/"
    # )

    # PPO2
    # model = PPO.load("log/PPO_6/best_model/best_model.zip", env=env)

    print(model.get_parameters())

    # model.learn(total_timesteps=1e7, callback=[checkpoint_callback, eval_callback])

# if __name__ == "__main__":
#     vec_env = SubprocVecEnv([lambda: make_env(False) for i in range(5)])
# model = PPO("CnnPolicy", env=env, verbose=1, tensorboard_log="./test_log/")
# model.learn(total_timesteps=1e7)
