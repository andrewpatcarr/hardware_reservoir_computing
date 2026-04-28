
import gymnasium as gym
from src.hardware_rc.dqn_rc import DQN_RC, DQNConfig

# env_name = "MountainCar-v0"
# env = gym.make(env_name)

agent = DQN_RC(watch=True)

agent.train()
