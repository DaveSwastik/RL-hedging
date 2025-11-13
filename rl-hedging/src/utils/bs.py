import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def bs_price_and_delta(S, K, r, q, sigma, tau, option_type='call'):
    """
    Black-Scholes European price and delta.
    S: spot
    K: strike
    r: risk-free rate (use 0 if not available)
    q: dividend yield (0)
    sigma: volatility (annualized)  => pass sqrt(v_t)
    tau: time to maturity in years
    returns: (price, delta)
    """
    if tau <= 0:
        # At maturity: payoff
        if option_type == 'call':
            price = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(K - S, 0.0)
            delta = -1.0 if S < K else 0.0
        return price, delta

    sigma = max(sigma, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type == 'call':
        price = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        delta = np.exp(-q * tau) * norm.cdf(d1)
    else:
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
        delta = -np.exp(-q * tau) * norm.cdf(-d1)
    return price, delta
