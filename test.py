import numpy as np
from stable_baselines3 import PPO

from portfolio_env import StockSelectionEnv
from utils import read_stock_data


# Create a function to initialize the environment with stock data and rebalancing frequency
def make_env(stock_data, rebalancing_frequency):
    def _init():
        return StockSelectionEnv(stock_data, rebalancing_frequency)

    return _init


# Sample financial data (replace with actual historical price data for multiple stocks)
stock_data = read_stock_data()

# Take 2nd half of the data for testing
stock_data = stock_data[len(stock_data) // 2:]

# Create a vectorized environment with 4 parallel instances, rebalancing quarterly
env = make_env(stock_data, 'monthly')()

# Create and train the PPO agent
model = PPO("MlpPolicy", env, verbose=2, device='auto')
model.load("ppo_stock_selection")

# Test the trained model
for i in range(len(stock_data)):
    stock_prices = stock_data['prc'].values[i]
    stock_actual_eps = stock_data['eps_actual'].values[i]
    earnings_to_price = stock_data['ni_me'].values[i]

    action, _states = model.predict(np.array((stock_prices, stock_actual_eps, earnings_to_price), dtype=np.float32), deterministic=True)

    # We use the environment for ease of rendering, but you can also render the model manually
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

# Save the model
model.save("ppo_stock_selection")
