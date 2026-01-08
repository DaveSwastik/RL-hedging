import unittest
import numpy as np
from src.envs.hedging_env import HedgingEnv

class TestHedgingEnv(unittest.TestCase):
    def setUp(self):
        self.sim_params = {
            'S0': 100, 'v0': 0.04, 'dt': 0.004, 'steps': 50, 'rho': -0.7,
            'regimes': [{'name': 'Test', 'theta': 0.04, 'kappa': 3.0, 'sigma_v': 0.2, 'mu': 0.05}],
            'trans_mat': [[1.0]]
        }
        self.option_spec = {'type': 'Asian', 'K': 100, 'maturity': 0.2, 'option_type': 'call'}
        
    def test_rebalancing_freq(self):
        # Test rebalancing every 5 steps
        env = HedgingEnv(self.sim_params, self.option_spec, trading_cost=0.0, rebalance_freq=5)
        env.reset()
        
        # Step 0: Can rebalance
        obs, _, _, _, _ = env.step(np.array([0.5]))
        self.assertEqual(env.position, 0.5, "Should rebalance at step 0")
        
        # Step 1: Cannot rebalance (should hold 0.5)
        obs, _, _, _, _ = env.step(np.array([1.0])) # Try to change to 1.0
        self.assertEqual(env.position, 0.5, "Should not rebalance at step 1")
        
        # Step 5: Can rebalance
        # Advance to step 5
        for _ in range(3): env.step(np.array([1.0]))
        
        # Now at step 5 (t_idx starts at 0, so after 5 steps we are at t_idx=5)
        # Wait, t_idx increments AFTER step.
        # t=0 (step called) -> t=1.
        # t=1 (step called) -> t=2.
        # ...
        # t=4 (step called) -> t=5.
        # Next call is with t=5. 5 % 5 == 0. Should rebalance.
        
        obs, _, _, _, _ = env.step(np.array([0.8]))
        self.assertEqual(env.position, 0.8, "Should rebalance at step 5")

    def test_drastic_change_penalty(self):
        env = HedgingEnv(self.sim_params, self.option_spec, trading_cost=0.0, rebalance_freq=1, drastic_change_penalty=10.0)
        env.reset()
        
        # Step 0: Change from 0.0 to 1.0. Delta = 1.0. Penalty = 10 * 1.0^2 = 10.0
        # Reward should be roughly -10.0 (ignoring hedging error for a moment, or assuming it's small)
        # Actually reward is -(hedging_error^2) - penalty.
        # Since it's not done, hedging_error is 0 (unless we changed reward structure).
        # In my implementation: reward = 0.0 - penalty.
        
        obs, reward, _, _, _ = env.step(np.array([1.0]))
        self.assertTrue(reward <= -10.0, f"Reward {reward} should include penalty of at least 10.0")

if __name__ == '__main__':
    unittest.main()
