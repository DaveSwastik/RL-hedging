# src/simulators/regime_heston.py
import numpy as np

class RegimeHestonSimulator:
    def __init__(self, params: dict, seed: int = None):
        """
        params: dict with keys:
          - dt, steps
          - S0, v0
          - regimes: list of regime dicts: {'theta','kappa','sigma_v','mu'}
          - trans_mat: n_reg x n_reg transition matrix
          - rho: correlation between price and variance Brownian increments
        """
        self.params = params
        self.rng = np.random.default_rng(seed)
        self._prepare()

    def _prepare(self):
        p = self.params
        self.dt = p.get("dt", 1/252)
        self.steps = p.get("steps", 50)
        self.S0 = p.get("S0", 100.0)
        self.v0 = p.get("v0", 0.04)
        self.n_regimes = len(p["regimes"])
        self.rho = p.get("rho", -0.7)

        # Pre-calculate regime parameters for vectorization
        self.thetas = np.array([r['theta'] for r in p['regimes']])
        self.kappas = np.array([r['kappa'] for r in p['regimes']])
        self.sigmas_v = np.array([r['sigma_v'] for r in p['regimes']])
        self.mus = np.array([r.get('mu', 0.0) for r in p['regimes']])
        self.trans_mat = np.array(p["trans_mat"])

    def simulate(self, n_paths: int, seed: int = None):
        """Generates a batch of paths."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        S = np.zeros((n_paths, self.steps + 1))
        v = np.zeros_like(S)
        regimes = np.zeros((n_paths, self.steps + 1), dtype=int)

        S[:, 0] = self.S0
        v[:, 0] = self.v0
        regimes[:, 0] = self.rng.integers(0, self.n_regimes, size=n_paths)

        for t in range(self.steps):
            # Draw correlated normals
            Z1 = self.rng.standard_normal((n_paths,))
            Z2 = self.rng.standard_normal((n_paths,))
            Wv = Z2
            Ws = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            # Regime transitions
            rand_uni = self.rng.uniform(size=(n_paths, 1))
            cum_probs = self.trans_mat[regimes[:, t]].cumsum(axis=1)
            regimes[:, t+1] = (rand_uni < cum_probs).argmax(axis=1)

            # Step variance and spot (Euler discretization with full truncation)
            idx = regimes[:, t]
            vt = np.maximum(v[:, t], 0)
            
            v[:, t+1] = (vt + self.kappas[idx] * (self.thetas[idx] - vt) * self.dt +
                         self.sigmas_v[idx] * np.sqrt(vt * self.dt) * Wv)
            
            S[:, t+1] = S[:, t] * np.exp((self.mus[idx] - 0.5 * vt) * self.dt +
                                         np.sqrt(vt * self.dt) * Ws)

        return S, v, regimes