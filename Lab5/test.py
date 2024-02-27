from racecar_gym.env_eval import RaceEnv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

env = RaceEnv(
    scenario="austria_competition_collisionStop",  # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
    render_mode="rgb_array_birds_eye",
    reset_when_collision=False,  # Only work for 'austria_competition' and 'austria_competition_collisionStop'
)

frames = []
last_steer = deque(maxlen=8)
obs, info = env.reset()
terminated = False
cnt = 0
x = []
y = []
while not terminated:
    now = env.render()
    frames.append(obs)

    observation = cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
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

    now_direction = np.clip(3 * (int(total / base) - 30) / 42, -1, 1)

    # dirkp = 1
    # direction = direction * (1 - dirkp) + now_direction * dirkp

    direction = round(now_direction, 6)

    # action, _states = model.predict(observation, deterministic=True)
    action = np.zeros(2, dtype=np.float32)
    if abs(direction) < 0.1:
        action[0] = 0.1
    elif abs(direction) < 0.2:
        action[0] = 0.05
    elif abs(direction) < 0.3:
        action[0] = 0.02
    else:
        action[0] = 0.01
    if cnt < 10:
        action[0] = 1

    last_steer.append(direction)
    for i in range(len(last_steer)):
        action[1] += last_steer[i] * 0.2

    cnt += 1

    obs, rew, terminated, truncated, info = env.step(action)
    # print(info["pose"][0], info["pose"][1])
    x.append(info["pose"][0])
    y.append(info["pose"][1])
    # plt.plot(x, y)
    # plt.savefig("temp/plot.png")
    print(
        "{:4d},{:.3f},{:.6f},{:.6f}".format(cnt, info["progress"], action[0], action[1])
    )
height, width, layers = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_path = "temp/video.mp4"
video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
for frame in frames:
    video.write(frame)
cv2.destroyAllWindows()
video.release()
