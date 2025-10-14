# src/baselines/dp_solver.py
import numpy as np
from numba import jit

# Numba is used to speed up the loop-intensive DP calculation
@jit(nopython=True)
def solve_dp_grid(steps, S_grid, A_grid, K, dt, sigma):
    """A simplified DP solver for a plain European option for demonstration."""
    n_S = len(S_grid)
    n_A = len(A_grid)
    
    # Value function V(t, s_idx) and Policy P(t, s_idx)
    V = np.zeros((steps + 1, n_S))
    
    # Terminal condition
    V[steps, :] = np.maximum(S_grid - K, 0)
    
    # Backward induction
    for t in range(steps - 1, -1, -1):
        for i_s in range(n_S):
            s = S_grid[i_s]
            best_value = -np.inf
            
            # Transition probabilities (simplified log-normal)
            # A full implementation would require integrating over the next state distribution
            mu_s_next = s * np.exp(-0.5 * sigma**2 * dt)
            sigma_s_next = s * np.sqrt(np.exp(sigma**2 * dt) - 1)
            
            # Find expected value for each action
            for i_a in range(n_A):
                # This part is highly simplified. A real DP solver needs to
                # correctly model the stochastic transitions and costs.
                # Here we just find the closest next state.
                s_next_expected = mu_s_next
                next_idx = np.abs(S_grid - s_next_expected).argmin()
                expected_future_val = V[t + 1, next_idx]
                
                if expected_future_val > best_value:
                    best_value = expected_future_val
            
            V[t, i_s] = best_value
            
    return V

class DPSolver:
    def __init__(self, option_spec, sim_params):
        self.option_spec = option_spec
        self.sim_params = sim_params

    def solve(self, n_S_bins=20, n_A_bins=10):
        print("Warning: DP Solver is a simplified toy example for small grids.")
        S_max = self.sim_params['S0'] * 2
        S_grid = np.linspace(1e-6, S_max, n_S_bins)
        A_grid = np.linspace(0, 1.0, n_A_bins) # delta positions
        
        # Using average volatility for simplicity
        avg_vol = np.mean([r['theta'] for r in self.sim_params['regimes']])
        
        value_grid = solve_dp_grid(
            self.sim_params['steps'],
            S_grid,
            A_grid,
            self.option_spec['K'],
            self.sim_params['dt'],
            np.sqrt(avg_vol)
        )
        return value_grid