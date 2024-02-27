import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym


class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env

		# create single env
		def envs():
			env = gym.make(config["env_id"], render_mode='rgb_array')
			env = gym.wrappers.ResizeObservation(env, (84, 84))
			env = gym.wrappers.GrayScaleObservation(env)
			env = gym.wrappers.FrameStack(env, 4)
			return env
		
		if self.envs_num>1:
			# create multiple envs
			env = gym.vector.AsyncVectorEnv([
				lambda: envs() for i in range(self.envs_num)
			])
		else:
			env = envs()

		self.env = env
		
		### TODO ###
		# initialize test_env
		t_env = gym.make(config["env_id"], render_mode='rgb_array')
		t_env = gym.wrappers.ResizeObservation(t_env, (84, 84))
		t_env = gym.wrappers.GrayScaleObservation(t_env)
		t_env = gym.wrappers.FrameStack(t_env, 4)
		self.test_env = t_env

		self.net = AtariNet(self.test_env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net

		observation = np.array(observation)
		# observation = torch.tensor(observation).cpu().unsqueeze(0).to(self.device)
		observation = torch.from_numpy(observation)
		if len(observation.shape) == 3:
			observation = observation.unsqueeze(0)
		observation = observation.to(self.device, dtype=torch.float32)

		if eval:
			with torch.no_grad():
				action, value, logp, _ = self.net(observation, eval=True)
		else:
			action, value, logp, _ = self.net(observation)
		
		return action, value, logp

		# if eval:
		# 	with torch.no_grad():
		# 		???, ???, ???, _ = self.net(observation, eval=True)
		# else:
		# 	???, ???, ???, _ = self.net(observation)
		
		# return NotImplementedError

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)

				### TODO ###
				# calculate loss and update network
				# ???, ???, ???, ??? = self.net(...)

				# action, value, logp, entropy = self.net(ob_train_batch)
				action, value, logp, entropy = self.net(ob_train_batch,False,ac_train_batch)

				# calculate policy loss
				# ratio = ???
				# surrogate_loss = ???

				ratio = torch.exp(logp - logp_pi_train_batch)
				rotio_A = ratio * adv_train_batch
				clip_rotio_A = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = -torch.mean(torch.min(rotio_A, clip_rotio_A))

				# calculate value loss
				# value_criterion = nn.MSELoss()
				# v_loss = value_criterion(...)

				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)
				
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				# total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	



