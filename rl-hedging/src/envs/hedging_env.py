# src/envs/hedging_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import norm
from src.simulators.regime_heston import RegimeHestonSimulator
from src.utils.payoffs import PAYOFF_FUNCTIONS

class HedgingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # --- FIX: Add trade_penalty to constructor ---
    def __init__(self, sim_params: dict, option_spec: dict, trading_cost: float = 0.0, trade_penalty: float = 0.02):
        super().__init__()
        self.sim_params = sim_params
        self.simulator = RegimeHestonSimulator(sim_params)
        
        self.option_spec = option_spec
        self.trading_cost = trading_cost
        self.trade_penalty_coeff = float(trade_penalty) # Configurable penalty
        
        # --- FIX: Unify payoff key and call signature ---
        opt_type_str = option_spec.get('type', 'Asian') 
        self.option_type_name = option_spec.get('option_type', 'call')
        self.payoff_fn = PAYOFF_FUNCTIONS[opt_type_str]
        # --- END FIX ---

        self.T_steps = sim_params['steps']
        self.K = self.option_spec.get('K', 100.0)
        self.T_maturity_years = self.T_steps * self.sim_params['dt'] 
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = 9
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # --- FIX: _bs_greeks handles calls and puts ---
    def _bs_greeks(self, S, K, T_rem, r, sigma, option_type='call'):
        if T_rem <= 1e-6 or sigma <= 1e-6:
            if option_type == 'call':
                return (1.0 if S > K else 0.0), 0.0
            else: # put
                return (-1.0 if S < K else 0.0), 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_rem) / (sigma * np.sqrt(T_rem))
        
        delta_call = norm.cdf(d1)
        if option_type == 'call':
            delta = delta_call
        else:
            delta = delta_call - 1.0
            
        vega = S * norm.pdf(d1) * np.sqrt(T_rem)
        return float(delta), float(vega)

    # --- FIX: _bs_price handles calls and puts ---
    def _bs_price(self, S, K, T_rem, sigma, option_type='call', r=0.0):
        if T_rem <= 1e-6 or sigma <= 1e-6:
            return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_rem) / (sigma * np.sqrt(T_rem))
        d2 = d1 - sigma * np.sqrt(T_rem)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T_rem) * norm.cdf(d2)
        else: # put
            price = K * np.exp(-r * T_rem) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return float(price)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        s0_range_min = self.K * 0.5
        s0_range_max = self.K * 1.5
        random_S0 = self.np_random.uniform(s0_range_min, s0_range_max)
        
        original_S0 = self.simulator.S0
        self.simulator.S0 = random_S0
        S, v, regimes = self.simulator.simulate(n_paths=1, seed=seed)
        self.simulator.S0 = original_S0
        
        self.S_path = S[0]
        self.v_path = v[0]
        
        # --- FIX: Add assert for path length ---
        assert len(self.S_path) >= self.T_steps + 1, f"Expected S_path length >= {self.T_steps+1}, got {len(self.S_path)}"

        self.t_idx = 0
        self.position = 0.0
        self.cash = 0.0
        
        self.running_sum = self.S_path[0]
        self.running_max = self.S_path[0]
        
        obs = self._get_obs()
        info = {
            'start_S0': self.S_path[0],
            'moneyness_start': self.S_path[0] / self.K
        }
        return obs, info

    def step(self, action):
        new_pos = float(action[0]) 
        
        # ---
        # MUST-FIX: Canonical P&L Accounting
        # (The redundant/incorrect lines from before are GONE)
        # ---
        
        current_price = self.S_path[self.t_idx]
        next_price = self.S_path[self.t_idx + 1]

        # 1. Execute trade at current_price
        trade_amount = new_pos - self.position
        self.cash -= trade_amount * current_price
        
        # 2. Pay transaction cost
        trade_cost = abs(trade_amount) * current_price * self.trading_cost
        self.cash -= trade_cost
        
        # 3. Update position (we now hold new_pos)
        self.position = new_pos
        
        # 4. Price moves. We earn P&L on the new position
        price_pnl = self.position * (next_price - current_price)
        self.cash += price_pnl
        # --- END FIX ---
        
        # Calculate Change in Option Value (the "target")
        T_rem_now = self.T_maturity_years - (self.t_idx * self.sim_params['dt'])
        T_rem_next = self.T_maturity_years - ((self.t_idx + 1) * self.sim_params['dt'])
        vol_now = np.sqrt(max(self.v_path[self.t_idx], 1e-8))
        vol_next = np.sqrt(max(self.v_path[self.t_idx+1], 1e-8))
        
        price_now = self._bs_price(current_price, self.K, T_rem_now, vol_now, self.option_type_name)
        price_next = self._bs_price(next_price, self.K, T_rem_next, vol_next, self.option_type_name)
        option_change = price_next - price_now
        
        step_hedging_error = (price_pnl - trade_cost) - option_change
        
        # Scaled Reward
        reward = -(step_hedging_error**2) / (self.K**2 + 1e-8)
        
        # --- FIX: Add configurable trade penalty ---
        trade_penalty = self.trade_penalty_coeff * (abs(trade_amount) / max(1.0, self.K))
        reward = reward - trade_penalty
        # --- END FIX ---

        self.t_idx += 1
        
        self.running_sum += self.S_path[self.t_idx]
        self.running_max = max(self.running_max, self.S_path[self.t_idx])
        
        done = (self.t_idx >= self.T_steps)
        info = {}

        if done:
            final_price = self.S_path[self.T_steps]
            
            # --- FIX: Explicit liquidation cost ---
            liquidation_cost = abs(self.position) * final_price * self.trading_cost
            self.cash -= liquidation_cost 
            final_pnl = self.cash + self.position * final_price 
            self.position = 0.0 # Explicitly set position to 0
            # --- END FIX ---
            
            payoff = self.payoff_fn(self.S_path, self.K, self.option_type_name)
            final_error = final_pnl - payoff
            
            info = {
                'payoff': payoff,
                'pnl': final_pnl,
                'hedging_error': final_error,
                'final_price': final_price,
                'start_S0': self.S_path[0],
                'moneyness_start': self.S_path[0] / self.K
            }

        obs = self._get_obs()
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        spot_price = self.S_path[self.t_idx]
        time_normalized = self.t_idx / self.T_steps
        
        moneyness = spot_price / self.K
        time_to_maturity_norm = (self.T_steps - self.t_idx) / self.T_steps
        T_rem_years = time_to_maturity_norm * self.T_maturity_years
        
        volatility = np.sqrt(max(self.v_path[self.t_idx], 1e-8))
        running_avg = self.running_sum / (self.t_idx + 1)
        
        # --- FIX: Call greeks with correct option type ---
        bs_delta, bs_vega = self._bs_greeks(spot_price, self.K, T_rem_years, 0.0, volatility, self.option_type_name)
        
        obs = np.array([
            time_normalized,
            moneyness,
            volatility,
            self.position,
            running_avg / self.K,
            self.running_max / self.K,
            time_to_maturity_norm,
            bs_delta,
            bs_vega
        ], dtype=np.float32)
        return obs