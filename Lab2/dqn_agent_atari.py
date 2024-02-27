import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# set a atari game
		env = gym.make(config["env_id"], render_mode='rgb_array')
		# resize screen to 84x84
		env = gym.wrappers.ResizeObservation(env, (84, 84))
		# convert to grayscale
		env = gym.wrappers.GrayScaleObservation(env)
		# stack 4 frames 
		env = gym.wrappers.FrameStack(env, 4)
		self.env = env

		### TODO ###
		# initialize test_env
		self.test_env = env

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	import numpy as np

	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection

		if random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			observation = np.array(observation)
			observation = torch.tensor(observation).unsqueeze(0).to(self.device)
			action = self.behavior_net(observation).argmax().item()
			

		return action

		# if random.random() < epsilon:
		# 	action = ???
		# else:
		# 	action = ???

		# return action

		# return NotImplementedError
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from behavior net
		# 3. calculate Q_target = r + gamma * Q(s', argmax_a Q(s',a))
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		# dqn
		q_value = self.behavior_net(state).gather(1, action.type(torch.int64))
		with torch.no_grad():
			q_next = self.target_net(next_state)
			q_target = reward + self.gamma * q_next.max(1)[0].unsqueeze(1) * (1 - done)

		# ddqn
		# q_value = self.behavior_net(state).gather(1, action.type(torch.int64))
		# with torch.no_grad():
		# 	q_next = self.behavior_net(next_state)
		# 	q_next_target = self.target_net(next_state)
		# 	q_next_max_action = q_next.argmax(dim=1, keepdim=True)
		# 	q_target = reward + self.gamma * q_next_target.gather(1, q_next_max_action).detach() * (1 - done)
		
		# print('q_value',q_value.shape)
		# print('q_next',q_next.shape)
		# print('q_target',q_target.shape)

		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		
		# q_value = ???
		# with torch.no_grad():
			# q_next = ???

			# if episode terminates at next_state, then q_target = reward
			# q_target = ???
        
		# criterion = ???
		# loss = criterion(q_value, q_target)

		# self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		# self.optim.zero_grad()
		# loss.backward()
		# self.optim.step()
	