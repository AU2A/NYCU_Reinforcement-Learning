from collections import OrderedDict
import gymnasium as gym
import numpy as np
from numpy import array, float32
from collections import deque
import cv2
import math

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
        test=False,
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
        # self.action_space = gym.spaces.box.Box(
        #     low=-1.0, high=1.0, shape=(2,), dtype=float32
        # )
        self.action_space = gym.spaces.MultiDiscrete([6, 9])
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
            low=0, high=255, shape=(8, 60, 60), dtype=np.uint8
        )
        # ----------------------------------

        #
        self.cur_step = 0

        # ----------------------------------
        self.test = test
        self.last_velocity = 0
        self.last_steering = 0
        self.last_progress = 0
        self.last_checkpoint = 0
        self.frames = deque(maxlen=N_frame)
        f = open("me/path.txt", "r")
        lines = f.readlines()
        f.close()
        self.x = []
        self.y = []
        for line in lines:
            self.x.append(float(line.split()[0]))
            self.y.append(float(line.split()[1]))

    # ----------------------------------

    def observation_postprocess(self, obs):
        # obs = obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)

        # ----------------------------------
        obs = obs[self.camera_name]
        obs = cv2.resize(obs, (60, 60))
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

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_nearest_points(self, x, y, target_x, target_y):
        distances = []
        for i in range(len(x)):
            distance = self.calculate_distance(x[i], y[i], target_x, target_y)
            distances.append(distance)

        # Sort distances in ascending order
        sorted_distances = sorted(distances)

        # Find the indices of the nearest two points
        nearest_indices = [
            distances.index(sorted_distances[0]),
            distances.index(sorted_distances[1]),
        ]

        # Get the nearest two points
        nearest_points = [
            (x[nearest_indices[0]], y[nearest_indices[0]]),
            (x[nearest_indices[1]], y[nearest_indices[1]]),
        ]

        return nearest_points

    def find_cut_line_point(self, nearest_points, target_x, target_y):
        x1, y1 = nearest_points[0]
        x2, y2 = nearest_points[1]

        if y1 == y2:
            return target_x, y1
        if x1 == x2:
            return x1, target_y

        # Calculate the slope of the line formed by the nearest two points
        slope = (y2 - y1) / (x2 - x1)

        # Calculate the slope of the perpendicular line passing through the target point
        perpendicular_slope = -1 / slope

        # Calculate the y-intercept of the perpendicular line
        perpendicular_y_intercept = target_y - perpendicular_slope * target_x

        # Calculate the x-coordinate of the cut line point
        cut_line_point_x = (y1 - perpendicular_y_intercept) / perpendicular_slope

        # Calculate the y-coordinate of the cut line point
        cut_line_point_y = (
            perpendicular_slope * cut_line_point_x + perpendicular_y_intercept
        )

        return cut_line_point_x, cut_line_point_y

    def step(self, actions):
        self.cur_step += 1
        # motor_action_idx, steering_action_idx = actions
        # motor_mapping = [0.01, 0.1, 0.2, 0.4, 0.7, 1]
        # motor_mapping = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
        motor_mapping = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        steering_mapping = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]

        # ----------------------------------
        motor_action = motor_mapping[actions[0]]
        steering_action = steering_mapping[actions[1]]
        # ----------------------------------

        # Add a small noise and clip the actions
        # motor_scale = 0.001
        # steering_scale = 0.01
        # motor_action = np.clip(
        #     motor_action + np.random.normal(scale=motor_scale), -1.0, 1.0
        # )
        # steering_action = np.clip(
        #     steering_action + np.random.normal(scale=steering_scale), -1.0, 1.0
        # )

        dict_actions = OrderedDict(
            [
                (self.motor_name, array(motor_action, dtype=float32)),
                (self.steering_name, array(steering_action, dtype=float32)),
            ]
        )
        obs, reward, terminated, truncated, info = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)

        # ----------------------------------

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
                self.last_progress = info["progress"]
                reward += 0.1
            reward += (
                motor_action
                - abs(motor_action - self.last_velocity)
                - abs(steering_action - self.last_steering)
            )
            now_x = info["pose"][0]
            now_y = info["pose"][1]
            nearest_points = self.find_nearest_points(self.x, self.y, now_x, now_y)
            cut_line_point = self.find_cut_line_point(nearest_points, now_x, now_y)
            reward -= self.calculate_distance(
                now_x, now_y, cut_line_point[0], cut_line_point[1]
            )

        self.last_velocity = motor_action
        self.last_steering = steering_action

        self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)
        # ----------------------------------

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
