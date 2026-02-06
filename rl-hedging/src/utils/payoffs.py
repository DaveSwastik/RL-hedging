# src/utils/payoffs.py
import numpy as np

def asian_option_payoff(S_path, K, option_type='call'):
    avg_price = np.mean(S_path)
    if option_type == 'call':
        return float(max(avg_price - K, 0.0))
    else: # put
        return float(max(K - avg_price, 0.0))

def lookback_option_payoff(S_path, K, option_type='call'):
    if option_type == 'call': # Fixed strike
        return float(max(np.max(S_path) - K, 0.0))
    else: # Fixed strike
        return float(max(K - np.min(S_path), 0.0))

# Add other payoffs (barrier, cliquet) here as needed.

PAYOFF_FUNCTIONS = {
    'asian': asian_option_payoff,    
    'lookback': lookback_option_payoff, 
}