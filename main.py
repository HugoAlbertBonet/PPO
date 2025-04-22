from finance_env import FinanceEnv
from ppo_optimized import PPO
from network import FeedForwardNN

env = FinanceEnv(data_dir="C:/Users/hugoa/MIARFID/0.TFM/data")
model = PPO(FeedForwardNN, env)
model.learn(1000000)