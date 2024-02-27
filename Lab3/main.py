from ppo_agent_atari import AtariPPOAgent

if __name__ == '__main__':

	config = {
		"gpu": True,
		"envs_num": 1,
		"training_steps": 2e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.4,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		# "logdir": 'log/Enduro-20231107-1/',
		"logdir": 'log/Video/',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		"eval_interval": 1e6,
		"eval_episode": 5,
	}
	agent = AtariPPOAgent(config)
	agent.load("log/Enduro-20231107-1/model_4000000_1845.pth")
	# agent.train()
	agent.evaluate_and_save_video()
