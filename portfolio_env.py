from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback


class StockSelectionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StockSelectionCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access custom data stored in the environment
        returns_deviation_mean = np.mean(self.model.env.get_attr('returns_deviation_mean'))
        returns_deviation_median = np.median(self.model.env.get_attr('returns_deviation_median'))
        rewards = np.mean(self.model.env.get_attr('rewards'))

        # Save the model
        self.model.save("ppo_stock_selection_temp")

        # Log the custom data to TensorBoard
        self.logger.record('rewards/returns_deviation_mean', returns_deviation_mean)
        self.logger.record('rewards/returns_deviation_median', returns_deviation_median)
        self.logger.record('rewards/rewards', rewards)

        return True


class StockSelectionEnv(gym.Env):
    def __init__(self, stock_data, rebalancing_frequency='monthly', min_stocks=50, max_stocks=100, render_mode='human'):
        super(StockSelectionEnv, self).__init__()

        self.render_mode: str = render_mode

        # Financial data (assumed to be a dictionary of pandas DataFrames, one for each stock)
        # Each DataFrame should have a 'price' column representing daily prices.
        self.stock_data: pd.DataFrame = stock_data
        self.num_stocks: int = len(stock_data['prc'].values[0])  # Number of stocks in the dataset
        self.min_stocks: int = min_stocks
        self.max_stocks: int = max_stocks

        self.max_steps: int = self.stock_data.shape[0] - 1
        self.current_step: int = 0
        self.stock_prices: np.ndarray = np.zeros((self.num_stocks,))
        self.stock_actual_returns: np.ndarray = np.zeros((self.num_stocks,))
        self.stock_actual_eps: np.ndarray = np.zeros((self.num_stocks,))

        self.returns_deviation: np.ndarray = np.zeros((self.num_stocks,))
        self.returns_deviation_mean: float = 0.0
        self.returns_deviation_median: float = 0.0
        self.rewards: float = 0.0

        # Define the rebalancing frequency
        if rebalancing_frequency == 'monthly':
            self.rebalance_interval = 1  # every month
        elif rebalancing_frequency == 'quarterly':
            self.rebalance_interval = 3  # every 3 months
        elif rebalancing_frequency == 'semiannually':
            self.rebalance_interval = 6  # every 6 months
        else:
            raise ValueError("Invalid rebalancing frequency. Choose from 'monthly', 'quarterly', 'semiannually'.")

        # Define observation space: expected returns or any other relevant features for all stocks
        # Stock prices + actual EPS + actual returns from the previous time step
        inf_box_space = spaces.Box(low=0, high=np.inf, shape=(self.num_stocks,), dtype=np.float32)
        bounded_box_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # Define action space: the expected returns of all stocks
        self.action_space = bounded_box_space

        self.observation_space = spaces.Dict({
            'stock_prices': inf_box_space,
            'stock_actual_eps': inf_box_space,
            'stock_actual_returns': bounded_box_space
        })

        self.reset()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple:  # type: ignore
        # Reset environment state
        self.current_step = 0
        self.rewards = 0.0
        self.stock_prices = self.stock_data['prc'].values[self.current_step]
        self.stock_actual_eps = self.stock_data['eps_actual'].values[self.current_step]
        self.stock_actual_returns = self.stock_data['stock_exret'].values[self.current_step]

        # Return initial observation (stock prices or features)
        return {
            'stock_prices': self.stock_prices,
            'stock_actual_eps': self.stock_actual_eps,
            'stock_actual_returns': self.stock_actual_returns
        }, {}

    def step(self, predicted_returns):
        terminated = False

        # Move forward in time by the rebalancing interval & the number of observations in the dataset
        self.current_step += self.rebalance_interval
        if self.current_step >= self.max_steps:
            self.current_step = self.max_steps
            terminated = True

        # Update stock prices after the time step
        new_prices = self.stock_data['prc'].values[self.current_step]
        new_actual_eps = self.stock_data['eps_actual'].values[self.current_step]
        new_actual_returns = self.stock_data['stock_exret'].values[self.current_step]

        self.returns_deviation = np.abs(predicted_returns - self.stock_actual_returns)

        # We don't want to penalize the agent for stocks that are not in the dataset (i.e. prices = 0)
        mask = new_prices != 0.0
        n_valid_stocks = np.sum(mask)

        # Calculate reward as a function of the calculated returns
        self.rewards = np.sum(np.exp(-6.0 * self.returns_deviation) * mask) / n_valid_stocks if n_valid_stocks > 0 else 0
        self.rewards = np.clip(self.rewards, -1, 1)  # Keep it in the range [-1, 1]

        # Logging
        self.returns_deviation_mean = np.mean(self.returns_deviation)
        self.returns_deviation_median = np.median(self.returns_deviation)

        # Update stock data
        self.stock_prices = new_prices
        self.stock_actual_eps = new_actual_eps
        self.stock_actual_returns = new_actual_returns

        # Observation: stock prices after the price change (i.e. prices for the current time step)
        observations = {
            'stock_prices': self.stock_prices,
            'stock_actual_eps': self.stock_actual_eps,
            'stock_actual_returns': self.stock_actual_returns
        }

        return observations, self.rewards, terminated, False, {}


def render(self):
    print(f'Step: {self.current_step}')
    print(f'Stock Prices: {self.stock_prices}')
    print(f'Stock Actual EPS: {self.stock_actual_eps}')
    print(f'Stock Actual Returns: {self.stock_actual_returns}')
    print(f'Average Predicted Returns: {self.predicted_returns_avg}')
    print(f'Average Rewards: {self.rewards}')
    print('-' * 20)
