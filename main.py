from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from portfolio_env import StockSelectionEnv, StockSelectionCallback
from utils import read_stock_data


# Create a function to initialize the environment with stock data and rebalancing frequency
def make_env(stock_data, rebalancing_frequency):
    def _init():
        return StockSelectionEnv(stock_data, rebalancing_frequency)

    return _init


# Sample financial data (replace with actual historical price data for multiple stocks)
stock_data = read_stock_data()

# Take first half of the data for training
stock_data = stock_data[:len(stock_data) // 2]

# Create a vectorized environment with 4 parallel instances, rebalancing quarterly
num_envs = 16
envs = DummyVecEnv([make_env(stock_data, 'monthly') for _ in range(num_envs)])

# Create and train the PPO agent
model = PPO("MultiInputPolicy", envs, n_steps=12, batch_size=32, gamma=0.992, clip_range=0.2, learning_rate=0.0003,
            verbose=2, device='auto', tensorboard_log="./ppo_stock_selection_tensorboard/")
model.learn(total_timesteps=100000, callback=StockSelectionCallback())

# Save the model
model.save("ppo_stock_selection")
