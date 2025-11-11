# RL Hedging: Path-dependent Options under Regime Shifts

This repository contains a reproducible codebase for training Reinforcement Learning agents to price and dynamically hedge path-dependent options in a market with regime shifts.

## Setup

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DaveSwastik/RL-hedging.git
   cd rl-hedging
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

## Project Structure

```
.
├── src/
│   ├── agents/         # RL agent implementations (PPO, A2C)
│   ├── analysis/       # Analysis and explanation tools
│   ├── baselines/      # Traditional hedging methods (Delta, DP, LSM)
│   ├── configs/        # Configuration files
│   ├── envs/          # RL environments for hedging
│   ├── simulators/     # Market simulators (Regime-switching Heston)
│   └── utils/         # Utility functions and payoffs
├── notebooks/         # Jupyter notebooks for experiments
├── models/           # Saved model weights
├── data/            # Historical price data
├── results/         # Experiment results
└── tests/           # Test suite
```

## Features

- Multiple RL algorithms (PPO, A2C) for option hedging
- Regime-switching Heston model for market simulation
- Traditional baselines (Delta hedging, Dynamic Programming, LSM)
- Historical data support with AAPL data
- Comprehensive evaluation metrics and backtesting
- Model interpretability tools

## Usage

### Training

The project provides multiple training approaches:

1. PPO Training:
   ```bash
   poetry run python src/agents/train.py
   ```

2. Custom A2C Training:
   ```bash
   poetry run python src/agents/train_custom_ac.py
   ```

3. Randomized PPO Training:
   ```bash
   poetry run python src/agents/train_ppo_randomized.py
   ```

### Evaluation

1. Run standard evaluation:
   ```bash
   poetry run python evaluation_script.py
   ```

2. Run backtesting:
   ```bash
   poetry run python backtesting_script.py
   ```

3. Run full evaluation suite:
   ```bash
   poetry run python run_evaluation_suite.py
   ```

### Notebooks

The project includes three Jupyter notebooks for detailed experimentation:

1. `01-simulators.ipynb`: Explore market simulation models
2. `02-train-ppo.ipynb`: Interactive PPO training walkthrough
3. `03-eval-and-explainability.ipynb`: Model evaluation and interpretability analysis

## Results

Results are stored in several directories:
- `results/`: Main experiment results (CSV format)
- `results_backtest/`: Backtesting results using historical data
- `results_suite/`: Comprehensive evaluation metrics
- `logs/`: Training logs and TensorBoard data for visualization

## Data

The project includes historical AAPL data for backtesting. You can download additional data using:
```bash
poetry run python download_data.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[MIT License] - see LICENSE file for details