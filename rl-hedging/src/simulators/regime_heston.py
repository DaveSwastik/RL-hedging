# src/simulators/regime_heston.py
import numpy as np
import warnings

class RegimeHestonSimulator:
    def __init__(self, params: dict, seed: int = None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self._prepare()
        self._validate_regimes() # New: Stability check

    def _prepare(self):
        p = self.params
        self.dt = float(p.get("dt", 1/252))
        self.sqrt_dt = np.sqrt(self.dt) # Optimization: pre-calculate
        self.steps = int(p.get("steps", 50))
        self.S0 = float(p.get("S0", 100.0))
        self.v0 = float(p.get("v0", 0.04))
        self.n_regimes = len(p["regimes"])
        self.rho = float(p.get("rho", -0.7))

        self.thetas = np.array([float(r['theta']) for r in p['regimes']])
        self.kappas = np.array([float(r['kappa']) for r in p['regimes']])
        self.sigmas_v = np.array([float(r['sigma_v']) for r in p['regimes']])
        self.mus = np.array([float(r.get('mu', 0.0)) for r in p['regimes']])
        self.trans_mat = np.array(p["trans_mat"])

    def _validate_regimes(self):
        """Ensures the simulation won't collapse due to config errors."""
        # Check transition matrix sums to ~1
        if not np.allclose(self.trans_mat.sum(axis=1), 1.0):
            warnings.warn("Transition matrix rows do not sum to 1. Normalizing...")
            self.trans_mat = self.trans_mat / self.trans_mat.sum(axis=1, keepdims=True)
            
        # Feller Condition Check: 2*kappa*theta > sigma_v^2
        for i in range(self.n_regimes):
            if 2 * self.kappas[i] * self.thetas[i] <= self.sigmas_v[i]**2:
                warnings.warn(f"Regime {i} violates Feller condition. Volatility may hit zero frequently.")

    def simulate(self, n_paths: int, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        S = np.zeros((n_paths, self.steps + 1))
        v = np.zeros_like(S)
        regimes = np.zeros((n_paths, self.steps + 1), dtype=int)

        S[:, 0] = self.S0
        v[:, 0] = self.v0
        # Start in a random regime or fixed if preferred
        regimes[:, 0] = self.rng.integers(0, self.n_regimes, size=n_paths)

        # Pre-calculate correlation factor
        rho_sqrt = np.sqrt(max(0.0, 1 - self.rho**2))

        for t in range(self.steps):
            Z1 = self.rng.standard_normal((n_paths,))
            Z2 = self.rng.standard_normal((n_paths,))
            Wv = Z2 * self.sqrt_dt
            Ws = (self.rho * Z1 + rho_sqrt * Z2) * self.sqrt_dt

            # Regime transitions
            rand_uni = self.rng.uniform(size=(n_paths, 1))
            cum_probs = self.trans_mat[regimes[:, t]].cumsum(axis=1)
            regimes[:, t+1] = (rand_uni < cum_probs).argmax(axis=1)

            idx = regimes[:, t]
            # Use Full Truncation: max(v, 0) for stability
            vt = np.maximum(v[:, t], 1e-8)

            # Variance step: Euler-Maruyama with safe square root
            v_next = vt + self.kappas[idx] * (self.thetas[idx] - vt) * self.dt + \
                     self.sigmas_v[idx] * np.sqrt(vt) * Wv
            v[:, t+1] = np.maximum(v_next, 1e-8)

            # Spot step: Exponential Euler
            S[:, t+1] = S[:, t] * np.exp((self.mus[idx] - 0.5 * vt) * self.dt + np.sqrt(vt) * Ws)

        # Apply Shocks (Logic remains the same, kept for consistency)
        try:
            shock_prob = float(self.params.get('shock_prob', 0.0))
            shock_scale = float(self.params.get('shock_scale', 0.0))
            if shock_prob > 0 and shock_scale > 0:
                for i in range(n_paths):
                    if self.rng.random() < shock_prob:
                        t_shock = self.rng.integers(1, self.steps + 1)
                        S[i, t_shock:] *= np.exp(-shock_scale)
        except (ValueError, TypeError):
            pass

        return S, v, regimes