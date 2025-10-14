# src/analysis/explain.py
import shap
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.envs.hedging_env import HedgingEnv

def explain_policy_with_shap(model_path, env: HedgingEnv, n_samples=100):
    """
    Uses SHAP to explain the agent's policy on a sample of states.
    """
    model = PPO.load(model_path)

    # 1. Collect a sample of observations from the environment
    obs_samples = []
    obs, _ = env.reset()
    obs_samples.append(obs)
    for _ in range(n_samples - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        obs_samples.append(obs)
    
    obs_samples = np.array(obs_samples)
    
    # 2. Create a SHAP explainer
    # SHAP needs a function that takes an array of observations and returns an array of actions
    def predict_fn(observations):
        return model.predict(observations, deterministic=True)[0]

    explainer = shap.KernelExplainer(predict_fn, obs_samples[:50]) # Use a smaller background set

    # 3. Compute SHAP values
    shap_values = explainer.shap_values(obs_samples)
    
    # 4. Plot the results
    obs_df = pd.DataFrame(obs_samples, columns=['time', 'spot', 'vol', 'pos', 'avg', 'max'])
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, obs_df, show=True)