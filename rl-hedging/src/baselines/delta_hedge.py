# src/baselines/delta_hedge.py
import numpy as np
from scipy.stats import norm

class DeltaHedger:
    def __init__(self, option_spec, trading_cost=0.0):
        self.K = option_spec['K']
        self.T = option_spec['maturity']
        self.option_type = option_spec.get('option_type', 'call')
        self.trading_cost = trading_cost

    def _bs_delta(self, S, K, t, T, r, sigma):
        if sigma <= 1e-6 or T <= t:
            return 1.0 if S > K else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else: # put
            return norm.cdf(d1) - 1.0

    def evaluate(self, S_path, v_path, dt):
        """Evaluates the delta hedging strategy on a single path."""
        n_steps = len(S_path) - 1
        position = 0.0
        cash = 0.0

        for t_idx in range(n_steps):
            t_rem = self.T - (t_idx * dt)
            S_t = S_path[t_idx]
            sigma_t = np.sqrt(v_path[t_idx])
            
            # For Asian options, a common approximation is to use BS on the underlying
            # A more sophisticated model would use a Turnbull-Wakeman approximation
            target_delta = self._bs_delta(S_t, self.K, t_idx * dt, self.T, 0.0, sigma_t)
            
            trade_amount = target_delta - position
            cash -= trade_amount * S_t
            cash -= abs(trade_amount) * S_t * self.trading_cost
            position = target_delta

        # Final liquidation
        final_price = S_path[-1]
        cash += position * final_price
        cash -= abs(position) * final_price * self.trading_cost
        
        return cash # Final P&L