import argparse
import json
import numpy as np
import requests
import cv2
import math
from collections import deque


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
    args = parser.parse_args()

    class Agent:
        def __init__(self):
            self.last_steer = deque(maxlen=5)
            self.last_steer_val = 0
            self.cnt = 0

        def act(self, observation):
            observation = cv2.cvtColor(
                observation.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY
            )
            observation = cv2.resize(observation, (84, 84))

            # if cnt % 10 == 0:
            #     cv2.imwrite("temp/temp.png", now)
            #     cv2.imwrite("temp/obs.png", observation)

            diridx = 0
            dirdepth = 0
            total = 0
            base = 1
            kp = 20000
            for i in range(84):
                now = min(60, (observation.transpose(1, 0)[i] == 255).sum())
                if now > dirdepth:
                    diridx = i
                    dirdepth = now
                total += now * i * (1 - math.sin(i * math.pi / 84)) * kp
                base += now * (1 - math.sin(i * math.pi / 84)) * kp

            now_direction = np.clip(3 * (int(total / base) - 42) / 42, -1, 1)

            # dirkp = 1
            # direction = direction * (1 - dirkp) + now_direction * dirkp

            direction = round(now_direction, 6)

            # action, _states = model.predict(observation, deterministic=True)
            action = np.zeros(2, dtype=np.float32)
            if abs(direction) < 0.1:
                action[0] = 0.2
            elif abs(direction) < 0.2:
                action[0] = 0.1
            elif abs(direction) < 0.3:
                action[0] = 0.05
            else:
                action[0] = 0.02
            action[0] = 0.3 * (1 - abs(direction)) + 0.01
            if self.cnt < 10:
                action[0] = 1

            self.last_steer.append(direction)
            for i in range(len(self.last_steer)):
                action[1] += self.last_steer[i] * 0.2

            # action[1] = self.last_steer_val * 0.5 + direction * 0.5
            # self.last_steer_val = action[1]

            self.cnt += 1
            print(action)
            return action

    # Initialize the RL Agent
    import gymnasium as gym

    agent = Agent()

    connect(agent, url=args.url)
