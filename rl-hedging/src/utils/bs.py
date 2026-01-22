import numpy as np
from scipy.stats import norm


def bs_greeks(S, K, r, q, sigma, tau, option_type='call'):
    """Calculates Price, Delta, Gamma, and Vega."""
    # Safety clamp for deep learning stability
    sigma = max(float(sigma), 1e-4)
    tau = max(float(tau), 1e-6)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # Common terms
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    # 1. Delta & Price
    if option_type == 'call':
        price = S * np.exp(-q * tau) * Nd1 - K * np.exp(-r * tau) * Nd2
        delta = np.exp(-q * tau) * Nd1
    else:  # Put
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
        delta = -np.exp(-q * tau) * norm.cdf(-d1)

    # 2. Gamma (Same for Call/Put)
    gamma = (np.exp(-q * tau) * nd1) / (S * sigma * np.sqrt(tau))

    # 3. Vega (Same for Call/Put)
    vega = S * np.exp(-q * tau) * nd1 * np.sqrt(tau)

    return price, delta, gamma, vega


# Backward compatibility wrapper
def bs_price_and_delta(S, K, r, q, sigma, tau, option_type='call'):
    p, d, _, _ = bs_greeks(S, K, r, q, sigma, tau, option_type)
    return p, d
