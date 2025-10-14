# src/baselines/lsm.py
import numpy as np
from sklearn.linear_model import Ridge

from src.utils.payoffs import PAYOFF_FUNCTIONS

class LSM_Pricer:
    """
    Prices a path-dependent option using LSM. This is more suited for American-style
    exercise but can be adapted to find the conditional expectation (price) at each step.
    """
    def __init__(self, simulator, option_spec, n_paths=10000, poly_degree=3):
        self.simulator = simulator
        self.option_spec = option_spec
        self.payoff_fn = PAYOFF_FUNCTIONS[option_spec['type']]
        self.n_paths = n_paths
        self.poly_degree = poly_degree

    def _build_features(self, S, running_avg):
        # Basis functions for regression
        return np.vstack([
            S**i * running_avg**j 
            for i in range(self.poly_degree + 1)
            for j in range(self.poly_degree + 1)
        ]).T

    def price(self):
        S, _, _ = self.simulator.simulate(n_paths=self.n_paths)
        
        # Calculate terminal payoffs
        payoffs = np.array([
            self.payoff_fn(path, self.option_spec['K'], self.option_spec.get('option_type', 'call'))
            for path in S
        ])
        
        cashflow = payoffs
        
        # Backward induction
        for t in range(self.simulator.steps - 1, 0, -1):
            St = S[:, t]
            running_avg = np.mean(S[:, :t+1], axis=1)
            
            # Use in-the-money paths for regression
            itm_mask = St > 0 # Use all paths for European style pricing
            
            X = self._build_features(St[itm_mask], running_avg[itm_mask])
            Y = cashflow[itm_mask] * np.exp(-0.0 * self.simulator.dt) # Assuming r=0 for simplicity
            
            if len(Y) < 10: # Not enough points to regress
                continue
                
            model = Ridge(alpha=1.0)
            model.fit(X, Y)
            
            # Predict continuation values for all paths
            X_all = self._build_features(St, running_avg)
            continuation_value = model.predict(X_all)
            
            # For European options, the price is the continuation value
            cashflow = continuation_value

        # Price at t=0 is the average of discounted cashflows
        return np.mean(cashflow * np.exp(-0.0 * self.simulator.dt))