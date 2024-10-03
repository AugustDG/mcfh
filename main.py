import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from portfolio_env import StockSelectionEnv, StockSelectionCallback
from utils import read_stock_data


# Create a function to initialize the environment with stock data and rebalancing frequency
def make_env(stock_data, rebalancing_frequency):
    def _init():
        return StockSelectionEnv(stock_data, rebalancing_frequency)

    return _init


def cosine_lr(progress, base_lr, min_lr=0.0):
    """
    Compute the learning rate using cosine annealing.

    Args:
        progress (float): The progress of the training, where 1 is the start and 0 is the end.
        base_lr (float): The initial (maximum) learning rate.
        min_lr (float): The minimum learning rate (default is 0).

    Returns:
        float: The computed learning rate.
    """
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * (1 - progress)))


# Sample financial data (replace with actual historical price data for multiple stocks)
stock_data = read_stock_data()

# Take all stock data until 2010 exclusively (by comparing the year of the index)
stock_data = stock_data.loc[stock_data.index.date < pd.Timestamp('2010-01-01').date()]

# Create a vectorized environment with 4 parallel instances, rebalancing quarterly
num_envs = 1024
envs = DummyVecEnv([make_env(stock_data, 'monthly') for _ in range(num_envs)])

# Create and train the PPO agent
model = PPO("MultiInputPolicy", envs, n_steps=12, batch_size=256, gamma=0.995, clip_range=0.2,
            learning_rate=lambda prg : cosine_lr(prg, 0.00004, 0.000001), ent_coef=0.0, n_epochs=5,
            verbose=2, device='cuda', tensorboard_log="./ppo_stock_selection_tensorboard/")

logging_cb = StockSelectionCallback()

model.save("ppo_stock_selection")

# Train the agent
model.learn(total_timesteps=10000000, callback=logging_cb)

# Save the model
model.save("ppo_stock_selection")
