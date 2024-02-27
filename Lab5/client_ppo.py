import argparse
import json
import numpy as np
import requests
from stable_baselines3 import PPO
from collections import deque
import cv2


def connect(agent, url: str = "http://localhost:5000"):
    while True:
        # Get the observation
        response = requests.get(f"{url}")
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break
        obs = json.loads(response.text)["observation"]
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f"{url}", json={"action": action_to_take.tolist()})
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break

        result = json.loads(response.text)
        terminal = result["terminal"]

        if terminal:
            print("Episode finished.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000",
        help="The url of the server.",
    )
    parser.add_argument(
        "--s",
        type=str,
        default="c",
        help=".",
    )
    args = parser.parse_args()

    class Agent:
        def __init__(self):
            if args.s == "a":
                self.model = PPO.load("log/PPO_6/best_model/best_model.zip")
                self.frames = deque(maxlen=8)
            else:
                self.model = PPO.load("log/PPO_circle/best_model/best_model.zip")
            self.cnt = 0

        def act(self, obs):
            self.cnt += 1
            if args.s == "a":
                obs = cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
                obs = cv2.resize(obs, (84, 84))
                while len(self.frames) < 7:
                    self.frames.append(obs)
                self.frames.append(obs)
                obs = np.stack(self.frames, axis=0)
                action = self.model.predict(obs, deterministic=True)
                if 400 < self.cnt and self.cnt < 600:
                    action[0][0] *= 0.1
                if 700 < self.cnt and self.cnt < 900:
                    action[0][0] *= 0.1
            else:
                obs = np.stack(obs, axis=0)
                action = self.model.predict(obs, deterministic=True)
            print(self.cnt, action[0])
            return action[0]

    # Initialize the RL Agent
    import gymnasium as gym

    agent = Agent()

    connect(agent, url=args.url)
