# src/envs/hedging_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.simulators.regime_heston import RegimeHestonSimulator
from src.utils.payoffs import PAYOFF_FUNCTIONS
from src.utils.bs import bs_price_and_delta # Import your BS pricer

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
        
        # --- *** THIS IS THE FIX *** ---
        # Force reward_scale to be a float, overriding any string from YAML
        default_scale = 1.0
        try:
            # Try to cast the value from the config (which might be a string)
            self.sim_params['reward_scale'] = float(self.sim_params.get('reward_scale', default_scale))
        except (ValueError, TypeError):
            # If it fails (e.g., it's an invalid string), use the default
            self.sim_params['reward_scale'] = default_scale
        # --- *** END FIX *** ---

        self.sim_params.setdefault('r', 0.0)

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
        # 1. Get positions and current state
        prev_pos = self.position
        new_pos = float(np.clip(action[0], -2.0, 2.0))
        trade_amount = new_pos - prev_pos
        
        current_price = float(self.S_path[self.t_idx])
        current_var = float(self.v_path[self.t_idx])
        tau_t = max(self.T - self.t_idx, 0) * self.sim_params.get('dt', 1/252)

        # 2. Calculate transaction costs
        tx_cost = abs(trade_amount) * current_price * self.trading_cost

        # 3. Get Option MTM at time t
        sigma_t = np.sqrt(max(current_var, 1e-8))
        price_t_raw, delta_t_raw = bs_price_and_delta(
            current_price, self.option_spec['K'],
            r=self.sim_params.get('r', 0.0), q=0.0,
            sigma=sigma_t, tau=tau_t, 
            option_type=self.option_spec.get('option_type','call')
        )
        price_t = float(price_t_raw)
        delta_t = float(delta_t_raw)


        # 4. Advance time
        self.t_idx += 1
        next_price = float(self.S_path[self.t_idx])
        next_var = float(self.v_path[self.t_idx])
        tau_tp1 = max(self.T - self.t_idx, 0) * self.sim_params.get('dt', 1/252)
        
        # 5. Get Option MTM at time t+1
        sigma_tp1 = np.sqrt(max(next_var, 1e-8))
        price_tp1_raw, delta_tp1_raw = bs_price_and_delta(
            next_price, self.option_spec['K'],
            r=self.sim_params.get('r', 0.0), q=0.0,
            sigma=sigma_tp1, tau=tau_tp1, 
            option_type=self.option_spec.get('option_type','call')
        )
        price_tp1 = float(price_tp1_raw)
        delta_tp1 = float(delta_tp1_raw)

        # 6. Calculate P&L components for this step
        d_option = price_tp1 - price_t  # Change in option value
        dS = next_price - current_price   # Change in stock value
        
        # P&L of hedger (who is SHORT the option and holds 'prev_pos' of stock)
        pnl_hedger = -d_option + (prev_pos * dS) - tx_cost

        # 7. Calculate Reward
        # Be defensive: reward_scale can come from user config (YAML/CLI) and
        # might be a string. Convert to float here and guard against zero.
        raw_scale = self.sim_params.get('reward_scale', 1.0)
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError):
            # Try to coerce common string formats, otherwise fallback to 1.0
            try:
                scale = float(str(raw_scale).replace(',', ''))
            except Exception:
                scale = 1.0
        if abs(scale) < 1e-12:
            scale = 1.0

        reward = pnl_hedger / scale
        
        # 8. Bookkeeping
        self.cash -= trade_amount * current_price
        self.cash -= tx_cost
        self.position = new_pos
        self.running_sum += next_price
        self.running_max = max(self.running_max, next_price)

        done = (self.t_idx >= self.T)
        info = {'pnl_hedger': pnl_hedger, 'd_option': d_option, 'tx_cost': tx_cost}

        if done:
            # Liquidate final position and calculate terminal P&L
            final_price = next_price
            self.cash += self.position * final_price
            self.cash -= abs(self.position) * final_price * self.trading_cost
            portfolio_value = self.cash
            
            payoff = self.payoff_fn(self.S_path, self.option_spec['K'], self.option_spec.get('option_type', 'call'))
            
            hedging_error = portfolio_value - payoff 
            
            # reuse the same safe `scale` computed above
            reward += hedging_error / scale
            
            info.update({'terminal_payoff': payoff, 'terminal_pnl': portfolio_value, 'terminal_hedge_err': hedging_error})

        obs = self._get_obs()
        terminated = done
        truncated = False
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