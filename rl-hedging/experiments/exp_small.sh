#!/bin/bash
# experiments/exp_small.sh

echo "--- Running Small Experiment: Training PPO Agent ---"
python src/agents/train.py --config_path src/configs/default.yaml --save_path models/ppo_asian_small.zip

echo "--- Experiment Finished ---"
echo "Model saved to models/ppo_asian_small.zip"
echo "To evaluate, use a Python script to call functions from src.analysis.metrics"