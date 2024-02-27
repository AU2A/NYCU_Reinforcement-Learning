import torch
import torch.nn as nn
import numpy as np
import os
import time
import cv2
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod


class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.envs_num = int(config["envs_num"])
		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": self.envs_num,
			})

		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError


	def train(self):
		episode_idx = 0
		observation, info = self.env.reset()
		episode_reward = [0] * self.envs_num
		episode_len = [0] * self.envs_num
		while self.total_time_step <= self.training_steps:
			action, value, logp_pi = self.decide_agent_actions(observation)
			action = action.cpu().detach().numpy()
			value = value.cpu().detach().numpy()
			logp_pi = logp_pi.cpu().detach().numpy()

			next_observation, reward, terminate, truncate, info = self.env.step(action)

			for i in range(self.envs_num):
				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				obs = {}
				obs["observation_2d"] = np.asarray(observation[i], dtype=np.float32)
				self.gae_replay_buffer.append(i, {
						"observation": obs,    # shape = (4,84,84)
						"action": np.array([action[i]]),      # shape = (1,)
						"reward": reward[i],      # shape = ()
						"value": value[i],        # shape = ()
						"logp_pi": logp_pi[i],    # shape = ()
						"done": terminate[i],     # shape = ()
					})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()
					break

			for i in range(self.envs_num):
				episode_reward[i] += reward[i]
				episode_len[i] += 1
			
			for i in range(self.envs_num):
				if terminate[i] or truncate[i]:
					if i==0:
						self.writer.add_scalar('Train/Episode Reward', episode_reward[0], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_len[0], self.total_time_step)
						print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]  episode: {self.total_time_step}  episode reward: {episode_reward}  episode len: {episode_len}")
					episode_reward[i] = 0
					episode_len[i] = 0
				
			observation = next_observation
			self.total_time_step += self.envs_num
				
			if self.total_time_step % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []

		for i in range(self.eval_episode):
			frames = []
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				# self.test_env.render()
				frames.append(self.test_env.render())
				action, _, _ = self.decide_agent_actions(observation)
				action = action.cpu().detach().numpy()
				next_observation, reward, terminate, truncate, info = self.test_env.step(action[0])
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	def evaluate_and_save_video(self):
		i=0
		all_rewards = []
		seeds = []
		while(i<5):
			frames = []
			seed=np.random.randint(1e8)
			observation, info = self.test_env.reset(seed=seed)
			total_reward = 0
			while True:
				frames.append(self.test_env.render())
				action, _, _ = self.decide_agent_actions(observation)
				action = action.cpu().detach().numpy()
				next_observation, reward, terminate, truncate, info = self.test_env.step(action[0])
				total_reward += reward
				if terminate or truncate:
					# print(f"episode {i} reward: {total_reward}")
					print(f"reward: {total_reward} seed: {seed}")
					# all_rewards.append(total_reward)
					break
				observation = next_observation

			if(total_reward>2300):
				i=i+1
				# print(f"episode {i} reward: {total_reward} seed: {seed}")
				all_rewards.append(total_reward)
				seeds.append(seed)
				height, width, layers = frames[0].shape
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				video_path = './video_'+str(int(total_reward))+'.mp4'
				video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
				for frame in frames:
					video.write(frame)
				cv2.destroyAllWindows()
				video.release()
		
		print("==============================================")
		print("Evaluating...")
		for i in range(5):
			print(f"episode {i+1} reward: {all_rewards[i]} seed: {seeds[i]}")
		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")

		# height, width, layers = frames[0].shape
		# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		# video_path = './video_'+str(int(total_reward))+'.mp4'
		# video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
		# for frame in frames:
		# 	video.write(frame)
		# cv2.destroyAllWindows()
		# video.release()

		# print(f"Video saved to {video_path}")
		# print("==============================================")
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()


	

