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
        accumulated_return_mean = np.mean(self.training_env.get_attr('accumulated_return'))
        rewards_mean = np.mean(self.locals['rewards'])
        num_selected_stocks = np.mean(self.training_env.get_attr('num_selected_stocks'))

        # Log the custom data to TensorBoard
        self.logger.record('rewards/accumulated_return_mean', accumulated_return_mean)
        self.logger.record('rewards/rewards_mean', rewards_mean)
        self.logger.record('rewards/num_selected_stocks', num_selected_stocks)

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
        self.num_selected_stocks: int = 0

        self.max_steps: int = self.stock_data.shape[0] - 1
        self.current_step: int = 0
        self.accumulated_return: float = 0
        self.stock_prices: np.ndarray = np.zeros((self.num_stocks,))
        self.stock_actual_eps: np.ndarray = np.zeros((self.num_stocks,))
        self.earnings_to_price: np.ndarray = np.zeros((self.num_stocks,))
        self.portfolio_weights: np.ndarray = np.zeros((self.num_stocks,))

        # Define the rebalancing frequency
        if rebalancing_frequency == 'monthly':
            self.rebalance_interval = 1  # every month
        elif rebalancing_frequency == 'quarterly':
            self.rebalance_interval = 3  # every 3 months
        elif rebalancing_frequency == 'semiannually':
            self.rebalance_interval = 6  # every 6 months
        else:
            raise ValueError("Invalid rebalancing frequency. Choose from 'monthly', 'quarterly', 'semiannually'.")

        # Define action space: binary vector of stock selection (0 or 1 for each stock)
        self.action_space = spaces.MultiBinary(self.num_stocks)

        # Define observation space: expected returns or any other relevant features for all stocks
        # Stock prices + actual EPS + earnings to price ratio + portfolio weights
        inf_box_space = spaces.Box(low=0, high=np.inf, shape=(self.num_stocks,), dtype=np.float32)
        ratio_box_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        binary_discrete_space = spaces.MultiBinary(self.num_stocks)
        self.observation_space = spaces.Dict({
            'stock_prices': inf_box_space,
            'stock_actual_eps': inf_box_space,
            'earnings_to_price': ratio_box_space,
            'portfolio_weights': binary_discrete_space
        })

        self.reset()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple:  # type: ignore
        # Reset environment state
        self.current_step = 0
        self.accumulated_return = 0
        self.stock_prices = self.stock_data['prc'].values[self.current_step]
        self.stock_actual_eps = self.stock_data['eps_actual'].values[self.current_step]
        self.earnings_to_price = self.stock_data['ni_me'].values[self.current_step]
        self.portfolio_weights = np.zeros((self.num_stocks,))

        # Return initial observation (stock prices or features)
        return {
            'stock_prices': self.stock_prices,
            'stock_actual_eps': self.stock_actual_eps,
            'earnings_to_price': self.earnings_to_price,
            'portfolio_weights': self.portfolio_weights
        }, {}

    def step(self, action):
        reward = 0
        truncated = False

        # Penalize the agent for choosing a stock with a 0 price & remove the stock from the portfolio
        if np.any(self.stock_prices[action == 1] == 0):
            reward -= 1

            # Remove the stock from the portfolio
            action[self.stock_prices == 0] = 0

        # Remove any excess stocks from the portfolio
        self.num_selected_stocks = int(np.sum(action))
        if self.num_selected_stocks > self.max_stocks:
            num_excess_stocks = self.num_selected_stocks - self.max_stocks
            reward -= num_excess_stocks

            # Remove the excess stocks from the portfolio

            # Get indices of the chosen stocks
            selected_stock_indices = np.where(action == 1)[0]
            # Randomly select the excess stocks to remove
            excess_stock_indices = np.random.choice(selected_stock_indices, (num_excess_stocks,), replace=False)
            action[excess_stock_indices] = 0

        # Reset the portfolio if the number of selected stocks is less than the minimum
        if self.num_selected_stocks < self.min_stocks:
            reward += -10
            truncated = True

            return self.reset(), reward, False, truncated, {}

        # Move forward in time by the rebalancing interval & the number of observations in the dataset
        self.current_step += self.rebalance_interval
        if self.current_step >= self.max_steps:
            self.current_step = self.max_steps

        # Update stock prices after the time step
        new_prices = self.stock_data['prc'].values[self.current_step]
        new_actual_eps = self.stock_data['eps_actual'].values[self.current_step]
        earnings_to_price = self.stock_data['ni_me'].values[self.current_step]

        self.portfolio_weights = action

        new_selected_prices = new_prices[self.portfolio_weights == 1]
        old_selected_prices = self.stock_prices[self.portfolio_weights == 1]

        # Calculate returns only for selected stocks (i.e. stocks with a weight of 1)
        selected_stock_returns = (new_selected_prices - old_selected_prices) / old_selected_prices

        # Calculate reward as the average return of the selected stocks
        average_return = np.mean(selected_stock_returns) if len(selected_stock_returns) > 0 else 0
        reward += average_return

        # Accumulate returns for informational purposes
        self.accumulated_return += average_return

        # Update stock prices
        self.stock_prices = new_prices
        self.stock_actual_eps = new_actual_eps
        self.earnings_to_price = earnings_to_price

        # Observation: stock prices after the price change (i.e. prices for the current time step)
        observations = {
            'stock_prices': self.stock_prices,
            'stock_actual_eps': self.stock_actual_eps,
            'earnings_to_price': self.earnings_to_price,
            'portfolio_weights': self.portfolio_weights
        }

        # Check if the episode is done (e.g., we run out of time)
        terminated = self.current_step >= self.stock_data.shape[0] - 1

        return observations, reward, terminated, truncated, {}

    def render(self):
        print(f'Step: {self.current_step}')
        print(f'Portfolio: {self.stock_data["prc"].columns[self.portfolio_weights == 1].values}')
        print(f'Stock Prices: {self.stock_prices}')
        print(f'Stock Actual EPS: {self.stock_actual_eps}')
        print(f'Earnings to Price: {self.earnings_to_price}')
        print(f'Accumulated Returns: {self.accumulated_return}')
        print('-' * 20)
