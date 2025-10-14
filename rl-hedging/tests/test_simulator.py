# tests/test_simulator.py
import yaml
import pytest
from src.simulators.regime_heston import RegimeHestonSimulator

@pytest.fixture
def default_config():
    """Loads the default configuration for the simulator."""
    with open('src/configs/default.yaml', 'r') as f:
        return yaml.safe_load(f)['simulator']

def test_simulator_initialization(default_config):
    """Tests if the simulator can be initialized without errors."""
    simulator = RegimeHestonSimulator(default_config)
    assert simulator is not None
    assert simulator.steps == default_config['steps']

def test_simulator_path_generation_shape(default_config):
    """Tests if the simulator produces paths of the correct shape."""
    simulator = RegimeHestonSimulator(default_config)
    n_paths = 10
    steps = default_config['steps']
    
    S, v, regimes = simulator.simulate(n_paths=n_paths)
    
    assert S.shape == (n_paths, steps + 1), "Spot price path shape is incorrect."
    assert v.shape == (n_paths, steps + 1), "Volatility path shape is incorrect."
    assert regimes.shape == (n_paths, steps + 1), "Regime path shape is incorrect."