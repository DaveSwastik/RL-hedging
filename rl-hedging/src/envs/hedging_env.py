# src/envs/hedging_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.simulators.regime_heston import RegimeHestonSimulator
from src.utils.payoffs import PAYOFF_FUNCTIONS

class HedgingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_params: dict, option_spec: dict, trading_cost: float = 0.0):
        super().__init__()
        self.sim_params = sim_params
        # Create a single-path simulator for the env step-by-step interaction
        self.simulator = RegimeHestonSimulator(sim_params)
        
        self.option_spec = option_spec
        self.trading_cost = trading_cost
        self.payoff_fn = PAYOFF_FUNCTIONS[option_spec['type']]
        self.T = sim_params['steps']

        # Action: continuous position in underlying (delta)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        # Observation: [time_norm, spot, vol, pos, running_avg, running_max]
        obs_dim = 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        S, v, regimes = self.simulator.simulate(n_paths=1, seed=seed)
        self.S_path = S[0]
        self.v_path = v[0]
        self.regimes = regimes[0]
        
        self.t_idx = 0
        self.position = 0.0
        self.cash = 0.0
        
        # Path-dependent stats
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
        
        # 2. Advance time
        self.t_idx += 1
        
        # 3. Update path stats
        self.running_sum += self.S_path[self.t_idx]
        self.running_max = max(self.running_max, self.S_path[self.t_idx])
        
        # 4. Determine reward and if episode is done
        done = (self.t_idx >= self.T)
        reward = 0.0
        info = {}

        if done:
            final_price = self.S_path[self.T]
            # Liquidate final position
            self.cash += self.position * final_price
            self.cash -= abs(self.position) * final_price * self.trading_cost
            self.position = 0.0
            
            portfolio_value = self.cash
            payoff = self.payoff_fn(self.S_path, self.option_spec['K'], self.option_spec.get('option_type', 'call'))
            
            hedging_error = portfolio_value - payoff
            reward = -(hedging_error ** 2) / 1e4 # Scale reward for stability
            
            info = {
                'payoff': payoff,
                'pnl': portfolio_value,
                'hedging_error': hedging_error,
                'final_price': final_price
            }

        obs = self._get_obs()
        terminated = done
        truncated = False # Not using time limits
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        time_normalized = self.t_idx / self.T
        spot_price = self.S_path[self.t_idx]
        volatility = np.sqrt(self.v_path[self.t_idx])
        
        running_avg = self.running_sum / (self.t_idx + 1)
        
        # Normalize features for better NN performance
        obs = np.array([
            time_normalized,
            spot_price / self.sim_params['S0'],
            volatility,
            self.position,
            running_avg / self.sim_params['S0'],
            self.running_max / self.sim_params['S0']
        ], dtype=np.float32)
        return obs

    def render(self, mode='human'):
        print(f"Step: {self.t_idx}, Price: {self.S_path[self.t_idx]:.2f}, Position: {self.position:.2f}, Cash: {self.cash:.2f}")