## 🎯 Model Saving & Loading Guide

### Where Models Are Saved

During training with `run_training.py`, models are saved **automatically** every 100 updates to:

```
./runs/training_run_01/
├── checkpoint_0.pth      (initial state, before training)
├── checkpoint_100.pth
├── checkpoint_200.pth
├── checkpoint_300.pth
├── ...
├── checkpoint_5000.pth   (final model after 5000 updates)
└── training_log.csv      (loss/metric logs)
```

### What's Saved in Each Checkpoint

Each `.pth` file contains:
```python
{
    'actor': actor.state_dict(),              # Actor network weights
    'critic': critic.state_dict(),            # Value critic weights
    'quantile_critic': quantile_critic.state_dict()  # CVaR/GPD critic weights (NEW!)
}
```

**IMPORTANT:** The quantile_critic was missing before and has now been added to the save!

---

### How to Load & Evaluate

After training completes, use `load_and_eval.py`:

```bash
# Run evaluation on the final model
python load_and_eval.py

# Or modify the script to test a specific checkpoint:
# checkpoint_path = './runs/training_run_01/checkpoint_2500.pth'
```

This will:
1. ✓ Load all 3 trained models from the checkpoint
2. ✓ Create an evaluation environment (WITHOUT shocks, for realistic testing)
3. ✓ Run 10 episodes and collect statistics
4. ✓ Print PnL, returns, and hedge error metrics

### Manual Loading

If you want to load a checkpoint in your own code:

```python
import torch
from src.agents.models import MlpActor, MlpCritic
from src.agents.cvar_ctitic import ExDrlCritic

# 1. Create fresh models (same architecture as training)
actor = MlpActor(obs_dim=6, hidden_dim=128, action_dim=1)
critic = MlpCritic(obs_dim=6, hidden_dim=128)
quantile_critic = ExDrlCritic(state_dim=6, hidden_dim=128, n_body_quantiles=32)

# 2. Load saved weights
checkpoint = torch.load('./runs/training_run_01/checkpoint_5000.pth', map_location='cpu')
actor.load_state_dict(checkpoint['actor'])
critic.load_state_dict(checkpoint['critic'])
quantile_critic.load_state_dict(checkpoint['quantile_critic'])

# 3. Set to eval mode
actor.eval()
critic.eval()
quantile_critic.eval()

# 4. Use for inference
with torch.no_grad():
    obs = torch.randn(1, 6)  # Your observation
    action_dist, _ = actor.step(obs, None)
    action = action_dist.sample()
```

---

### Training Progress Monitoring

While training runs, monitor the CSV log:

```bash
# Watch training in real-time
tail -f ./runs/training_run_01/training_log.csv

# Key columns to watch:
# - actor_loss: Should decrease
# - value_loss: Should decrease
# - gpd_loss: Should be > 0 (not frozen!)
# - tail_percent: Should be ~0.05 (5% tail detection)
# - entropy: Should decay as training progresses
```

---

### File Structure After Training

```
runs/training_run_01/
├── checkpoint_0.pth          (initial)
├── checkpoint_100.pth        
├── checkpoint_200.pth        
├── ...
├── checkpoint_5000.pth       ← Use this one for evaluation
└── training_log.csv
```

**Recommended:** Use the final checkpoint (5000 or higher update number) for best performance.

---

### Common Use Cases

#### 1. **Best Model Selection**
```bash
# Load and check different checkpoints
python load_and_eval.py  # Modify checkpoint_path inside script
```

#### 2. **Continuing Training**
```python
# Load from checkpoint and resume
checkpoint = torch.load('./runs/training_run_01/checkpoint_2500.pth')
actor.load_state_dict(checkpoint['actor'])
# ... resume training loop
```

#### 3. **Production Deployment**
```python
# Load the final trained model
actor, critic, quantile_critic = load_checkpoint(
    './runs/training_run_01/checkpoint_5000.pth',
    device='cpu'
)
# Use actor for live predictions
```

---

### Troubleshooting

**Q: Where are my models?**
- A: They're in `./runs/training_run_01/checkpoint_*.pth`

**Q: Why is quantile_critic missing when I load?**
- A: Old checkpoints (before this fix) don't have it. Retrain with the updated code.

**Q: How do I load a model with GPU?**
- A: `load_checkpoint(checkpoint_path, device='cuda')`

**Q: Can I load actor only without critic?**
- A: Yes, just load `checkpoint['actor']` into your actor model.
