# src/envs/historical_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.payoffs import PAYOFF_FUNCTIONS

class HistoricalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, historical_data_path: str, option_spec: dict, sim_params: dict, trading_cost: float = 0.0, vol_window=20):
        super().__init__()

        # Load historical prices
        self.df = pd.read_csv(historical_data_path, index_col='Date', parse_dates=True)
        self.prices = self.df['Close'].values
        
        # Calculate log returns for volatility estimation
        self.log_returns = np.log(self.prices[1:] / self.prices[:-1])

        self.option_spec = option_spec
        self.trading_cost = trading_cost
        self.payoff_fn = PAYOFF_FUNCTIONS[option_spec['type']]
        self.T_steps = sim_params['steps'] # Episode length (e.g., 50 days)
        self.dt = sim_params['dt'] # Time step size (for BS delta calculation)
        self.S0_sim = sim_params['S0'] # S0 from config for normalization reference

        self.vol_window = vol_window # Window for rolling volatility estimation

        if len(self.prices) <= self.T_steps + self.vol_window:
            raise ValueError(f"Historical data ({len(self.prices)} points) too short for episode length ({self.T_steps}) + vol window ({self.vol_window}).")

        # Action: continuous position in underlying (delta)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        # Observation: [time_norm, spot_norm, vol, pos, running_avg_norm, running_max_norm]
        obs_dim = 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _estimate_volatility(self, current_step_index):
        """Estimates volatility at the current step using past returns."""
        # Ensure we don't look ahead and have enough data
        start_idx = max(0, current_step_index - self.vol_window)
        if current_step_index == 0 or start_idx >= len(self.log_returns):
            # Return a default low vol if no history available
            # Use annualized standard deviation
            return 0.1 / np.sqrt(252) # Daily std dev for 10% annual vol

        past_returns = self.log_returns[start_idx:current_step_index]
        if len(past_returns) < 2:
             return 0.1 / np.sqrt(252)

        # Calculate daily standard deviation from log returns
        daily_std_dev = np.std(past_returns)
        return daily_std_dev


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random episode start point from historical data
        # Ensure we have enough data for the episode AND the initial vol window
        self.start_idx = self.np_random.integers(self.vol_window, len(self.prices) - self.T_steps - 1)
        self.current_global_idx = self.start_idx

        # Extract the 51-day price path for this episode
        self.S_path = self.prices[self.start_idx : self.start_idx + self.T_steps + 1]

        # Initialize estimated volatility path for this episode (will be updated step-by-step)
        self.v_path_est = np.zeros(self.T_steps + 1) # Store estimated daily variances
        self.v_path_est[0] = self._estimate_volatility(self.start_idx)**2 / self.dt # Approximation for Heston variance

        # Reset agent state
        self.t_idx = 0 # Step within the episode (0 to T_steps)
        self.position = 0.0
        self.cash = 0.0

        # Path-dependent stats for the option (recalculated for each episode)
        self.running_sum = self.S_path[0]
        self.running_max = self.S_path[0]

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # 1. Execute trade based on agent's desired position
        new_pos = float(action[0])
        trade_amount = new_pos - self.position
        current_price = self.S_path[self.t_idx]

        self.cash -= trade_amount * current_price
        self.cash -= abs(trade_amount) * current_price * self.trading_cost
        self.position = new_pos

        # 2. Advance time within the episode
        self.t_idx += 1
        self.current_global_idx += 1

        # 3. Update path stats
        self.running_sum += self.S_path[self.t_idx]
        self.running_max = max(self.running_max, self.S_path[self.t_idx])

        # 4. Estimate volatility for the NEXT step's observation
        # Note: Volatility uses info up to the end of the current step (t_idx)
        # We use current_global_idx which points to the price at the *end* of step t_idx
        current_daily_std_dev = self._estimate_volatility(self.current_global_idx)
        self.v_path_est[self.t_idx] = current_daily_std_dev**2 / self.dt # Store estimated variance

        # 5. Determine reward and if episode is done
        done = (self.t_idx >= self.T_steps)
        reward = 0.0 # Standard RL hedging often uses only terminal reward
        info = {}

        if done:
            final_price = self.S_path[self.T_steps]
            # Liquidate final position
            self.cash += self.position * final_price
            self.cash -= abs(self.position) * final_price * self.trading_cost
            self.position = 0.0

            portfolio_value = self.cash
            payoff = self.payoff_fn(self.S_path, self.option_spec['K'], self.option_spec.get('option_type', 'call'))

            hedging_error = portfolio_value - payoff
            # Reward is negative squared error (as in training)
            # You can scale this reward if needed for training stability, but for eval it's fine
            reward = -(hedging_error ** 2) / 1e4 # Scale reward

            info = {
                'payoff': payoff,
                'pnl': portfolio_value,
                'hedging_error': hedging_error,
                'final_price': final_price,
                's_path': self.S_path, # Include path for analysis if needed
            }

        obs = self._get_obs()
        terminated = done
        truncated = False # Not using time limits

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        time_normalized = self.t_idx / self.T_steps
        spot_price = self.S_path[self.t_idx]
        # Use the estimated volatility (sqrt of variance)
        volatility = np.sqrt(max(self.v_path_est[self.t_idx], 1e-8)) # Ensure non-negative before sqrt

        running_avg = self.running_sum / (self.t_idx + 1)

        # Normalize features relative to the STARTING price of the EPISODE
        # Or relative to S0 from config for consistency with training? Let's use S0_sim.
        obs = np.array([
            time_normalized,
            spot_price / self.S0_sim,
            volatility,
            self.position,
            running_avg / self.S0_sim,
            self.running_max / self.S0_sim
        ], dtype=np.float32)
        return obs

    def render(self, mode='human'):
        print(f"Step: {self.t_idx}, Price: {self.S_path[self.t_idx]:.2f}, Position: {self.position:.2f}, Cash: {self.cash:.2f}")