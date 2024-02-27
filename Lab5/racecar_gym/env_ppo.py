from collections import OrderedDict
import gymnasium as gym
import numpy as np
from numpy import array, float32
from collections import deque
import cv2

# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api


class RaceEnv(gym.Env):
    camera_name = "camera_competition"
    motor_name = "motor_competition"
    steering_name = "steering_competition"
    """The environment wrapper for RaceCarGym.
    
    - scenario: str, the name of the scenario.
        'austria_competition' or
        'plechaty_competition'
    
    Notes
    -----
    - Assume there are only two actions: motor and steering.
    - Assume the observation is the camera value.
    """

    def __init__(
        self,
        scenario: str,
        render_mode: str = "rgb_array_birds_eye",
        reset_when_collision: bool = True,
        N_frame=8,
        **kwargs,
    ):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f"SingleAgent{self.scenario}-v0"
        self.env = gym.make(
            id=self.env_id,
            render_mode=render_mode,
            reset_when_collision=reset_when_collision,
            **kwargs,
        )
        self.render_mode = render_mode
        # Assume actions only include: motor and steering
        self.action_space = gym.spaces.box.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=float32
        )
        # Assume observation is the camera value
        # noinspection PyUnresolvedReferences
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert (
            self.camera_name in observation_spaces
        ), f"One of the sensors must be {self.camera_name}. Check the scenario file."
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=255, shape=(3, 128, 128), dtype=np.uint8
        # )

        # ----------------------------------
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(8, 84, 84), dtype=np.uint8
        )
        # ----------------------------------

        self.cur_step = 0

        # ----------------------------------
        self.last_velocity = 0
        self.last_steering = 0
        self.last_progress = 0
        self.last_checkpoint = 0
        self.frames = deque(maxlen=N_frame)

    def observation_postprocess(self, obs):
        # obs = obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)

        # ----------------------------------
        obs = obs[self.camera_name]
        obs = cv2.resize(obs, (84, 84))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # ----------------------------------

        return obs

    def reset(self, *args, **kwargs: dict):
        self.cur_step = 0
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.observation_postprocess(obs)
        # ----------------------------------
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)
        # ----------------------------------
        return obs, *others

    def step(self, actions):
        self.cur_step += 1
        motor_action, steering_action = actions
        if 400 < self.cur_step and self.cur_step < 580:
            motor_action *= 0.2
        if 700 < self.cur_step and self.cur_step < 880:
            motor_action *= 0.2

        # Add a small noise and clip the actions
        motor_scale = 0.01
        steering_scale = 0.1
        motor_action = np.clip(
            motor_action + np.random.normal(scale=motor_scale), -1.0, 1.0
        )
        steering_action = np.clip(
            steering_action + np.random.normal(scale=steering_scale), -1.0, 1.0
        )

        dict_actions = OrderedDict(
            [
                (self.motor_name, array(motor_action, dtype=float32)),
                (self.steering_name, array(steering_action, dtype=float32)),
            ]
        )
        obs, reward, terminated, truncated, info = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)

        if info["time"] > 100:
            terminated = True
        elif info["wall_collision"]:
            reward = -50
            terminated = True
        elif info["wrong_way"]:
            reward = -50
            terminated = True
        else:
            reward = 0
            if info["progress"] > self.last_progress + 0.01:
                if info["progress"] > 0.36:
                    reward += (info["progress"] - 0.36) * 10
                self.last_progress = info["progress"]
                reward += 0.1
            reward += (
                motor_action
                - abs(motor_action - self.last_velocity)
                - abs(steering_action - self.last_steering)
            )

        self.last_velocity = motor_action
        self.last_steering = steering_action
        # ----------------------------------
        self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)
        # ----------------------------------
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
