import pandas as pd
from stable_baselines3 import PPO

from portfolio_env import StockSelectionEnv
from utils import read_stock_data

# Sample financial data (replace with actual historical price data for multiple stocks)
stock_data = read_stock_data()

# Take everything from 2010 onwards
stock_data = stock_data.loc[stock_data.index.date >= pd.Timestamp('2010-01-01').date()]

# Create a vectorized environment with 4 parallel instances, rebalancing quarterly
env = StockSelectionEnv(stock_data, rebalancing_frequency='monthly')

# Create and train the PPO agent
model = PPO("MultiInputPolicy", env, verbose=2, device='auto')
model.load("ppo_stock_selection")

# Test the trained model
for i in range(len(stock_data)):
    stock_prices = stock_data['prc'].values[i]
    stock_actual_eps = stock_data['eps_actual'].values[i]
    stock_actual_returns = stock_data['stock_exret'].values[i]

    observations = {
        'stock_prices': stock_prices,
        'stock_actual_eps': stock_actual_eps,
        'stock_actual_returns': stock_actual_returns
    }

    action, _states = model.predict(observations, deterministic=True)

    # We use the environment for ease of rendering, but you can also render the model manually
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}")
    print(f"Reward: {reward}")
    #env.render()

# Save the model
model.save("ppo_stock_selection")
