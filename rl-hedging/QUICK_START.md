## 📁 MODEL SAVING FLOWCHART

```
┌─────────────────────────────────────────────────────────────────┐
│  poetry run python run_training.py                              │
│  (Training starts)                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
                    Creates ./runs/training_run_01/
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ↓                    ↓                    ↓
   training_log.csv   checkpoint_0.pth    Every 100 updates:
   (loss metrics)     (initial model)      save checkpoint_*.pth
        │                    │                    │
        │                    │        checkpoint_100.pth ─┐
        │                    │        checkpoint_200.pth  │
        │                    │        checkpoint_300.pth  │ All 3 models:
        │                    │        ...                 │ - actor
        │                    │        checkpoint_5000.pth │ - critic
        │                    │                            │ - quantile_critic
        │                    └────────────────────────────┘
        │
        └───── Use for monitoring────→ tail -f training_log.csv
                  progress             Watch tail_percent ~0.05
                                      Watch gpd_loss > 0
```

---

## 📋 QUICK START: TRAIN → EVALUATE

### Step 1: Start Training
```bash
cd c:\Users\svast\OneDrive\Desktop\RL-hedging\rl-hedging
poetry run python run_training.py
```

**Output:**
```
Starting training run: 5000 updates
Logging to: ./runs/training_run_01/training_log.csv
With shocks: shock_prob=0.05, shock_scale=1.0
Reward scale: 100.0x

✓ Models will be saved to: ./runs/training_run_01/
  - checkpoint_0.pth (initial)
  - checkpoint_100.pth, checkpoint_200.pth, ... (every 100 updates)
  - checkpoint_5000.pth (final)

[Training proceeds...]
Update 0 stats: {...}
Update 10 stats: {...}
...
```

---

### Step 2: Monitor Training (in another terminal)
```bash
tail -f ./runs/training_run_01/training_log.csv
```

Watch for:
- `tail_percent` → should be ~0.05
- `gpd_loss` → should be > 0 (not frozen)
- `actor_loss` → should decrease
- `entropy` → should decay

---

### Step 3: After Training Completes
```bash
python load_and_eval.py
```

**Output:**
```
✓ Loaded checkpoint from: ./runs/training_run_01/checkpoint_5000.pth
  - Actor: 18945 params
  - Critic: 18817 params
  - Quantile Critic: 19345 params
✓ Loaded quantile_critic from checkpoint

Running evaluation (10 episodes, no shocks)...

Episode 1/10: PnL=0.0342, Return=0.0128
Episode 2/10: PnL=0.0156, Return=0.0089
...

============================================================
EVALUATION STATISTICS:
============================================================
avg_pnl              :   0.024567
std_pnl              :   0.008942
min_pnl              :   0.005123
max_pnl              :   0.041234
avg_return           :   0.009234
std_return           :   0.004521
avg_hedge_error      :   0.001234
std_hedge_error      :   0.000567
============================================================
```

---

## 🎯 KEY FILES

| File | Purpose |
|------|---------|
| `run_training.py` | Main training script (saves models every 100 updates) |
| `load_and_eval.py` | Load checkpoint & evaluate on new episodes |
| `./runs/training_run_01/checkpoint_*.pth` | Saved model weights |
| `./runs/training_run_01/training_log.csv` | Training metrics log |
| `src/agents/trainer_a2c.py` | Trainer with model saving code |

---

## 💾 WHAT'S IN A CHECKPOINT

Each `.pth` file (PyTorch checkpoint) contains:

```python
{
    'actor': {
        'net.0.weight': tensor([...]),
        'net.0.bias': tensor([...]),
        'net.2.weight': tensor([...]),
        ...
    },
    'critic': {
        'net.0.weight': tensor([...]),
        'net.0.bias': tensor([...]),
        ...
    },
    'quantile_critic': {
        'net.0.weight': tensor([...]),
        'net.0.bias': tensor([...]),
        ...
    }
}
```

All three models are restored when you load a checkpoint!

---

## ✅ CHECKLIST

Before running training:
- [ ] `poetry install` (dependencies installed)
- [ ] `src/configs/default.yaml` exists (or fallback used)
- [ ] `./runs/` directory writable

After training:
- [ ] `./runs/training_run_01/checkpoint_5000.pth` exists
- [ ] `./runs/training_run_01/training_log.csv` has data
- [ ] Run `python load_and_eval.py` to test the model
- [ ] Check metrics look reasonable (PnL, returns, hedge errors)
