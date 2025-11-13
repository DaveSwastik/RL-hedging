# Clean Terminal Output with tqdm

## What Changed?

**Before:** Messy terminal with debug output every update
```
Update 10 stats: {'actor_loss': 2.451, 'value_loss': 0.892, ...}
Update 20 stats: {'actor_loss': 2.234, 'value_loss': 0.789, ...}
=== GPD Loss Debug (Update) ===
    detached_returns: min=-0.001234, max=0.005678, ...
    [WARNING] u_flat=-0.000123 outside return range [...]
    [FALLBACK] Using empirical 5.0%-percentile: -0.000456
    tail_count: 45 / 800 (5.63%)
    u compared to return stats:
      - returns < u_min? 123 samples
      - returns < u_mean? 456 samples
      - returns < u_max? 789 samples
... (repeats 500 times!)
```

**After:** Clean progress bar with key metrics at 10% intervals
```
Training: 30%|████████▌                   | 1500/5000 [05:30<12:45, Actor Loss: 0.0542 | Value Loss: 0.0234 | GPD Loss: 0.3421 | Tail Events: 58,234
```

---

## What's Displayed?

At **every 10% of training completion**, you see:
- **Actor Loss**: How well the policy is learning
- **Value Loss**: How well the value function estimates returns
- **GPD Loss**: How well the tail risk model is trained
- **Tail Events**: Total count of tail samples detected so far

---

## How to Use

### Default (Clean Output)
```bash
poetry run python run_training.py
```

**Output:**
```
Training: 100%|██████████| 5000/5000 [2:15:00<00:00, Actor Loss: 0.0245 | Value Loss: 0.0089 | GPD Loss: 0.5621 | Tail Events: 125,680

================================================================================
TRAINING COMPLETE ✓
================================================================================
Total updates: 5000
Time elapsed: 2.25 hours
Model saved: ./runs/training_run_01/checkpoint_final.pth
Log saved: ./runs/training_run_01/training_log.csv
================================================================================
```

### Enable Verbose Debug (if needed)
If you want to see all the detailed GPD debugging again:

```python
# In run_training.py, after creating trainer:
trainer.verbose = True  # Enable debug output
```

Then run:
```bash
poetry run python run_training.py
```

**Output:** Full debug output every update (original behavior)

---

## Key Metrics Explained

| Metric | Meaning | Good Range | How to Interpret |
|--------|---------|------------|------------------|
| **Actor Loss** | Policy gradient loss | Decreases over time | Should trend downward |
| **Value Loss** | MSE between pred & actual returns | Decreases over time | Should trend downward |
| **GPD Loss** | Tail risk modeling loss | Non-zero | > 0 means tail model is learning |
| **Tail Events** | Cumulative tail samples found | ~5% of total | Should see consistent growth |

---

## Example Training Run Output

```
Training:   0%|                             | 0/5000 [00:00<?, ?it/s]
Training:  10%|██▌                          | 500/5000 [00:45<06:45] Actor Loss: 2.1234 | Value Loss: 0.5678 | GPD Loss: 0.0234 | Tail Events: 12,345
Training:  20%|█████                        | 1000/5000 [01:30<05:00] Actor Loss: 1.8932 | Value Loss: 0.4521 | GPD Loss: 0.1456 | Tail Events: 24,567
Training:  30%|██████▌                      | 1500/5000 [02:15<04:15] Actor Loss: 1.5621 | Value Loss: 0.3890 | GPD Loss: 0.2789 | Tail Events: 37,234
Training:  40%|█████████                    | 2000/5000 [03:00<03:30] Actor Loss: 1.2345 | Value Loss: 0.3123 | GPD Loss: 0.3456 | Tail Events: 49,876
Training:  50%|███████████▌                 | 2500/5000 [03:45<03:45] Actor Loss: 0.9876 | Value Loss: 0.2456 | GPD Loss: 0.4123 | Tail Events: 62,543
Training:  60%|██████████████               | 3000/5000 [04:30<03:00] Actor Loss: 0.7234 | Value Loss: 0.1890 | GPD Loss: 0.4567 | Tail Events: 75,123
Training:  70%|█████████████████▌           | 3500/5000 [05:15<02:15] Actor Loss: 0.5123 | Value Loss: 0.1456 | GPD Loss: 0.5012 | Tail Events: 87,654
Training:  80%|████████████████████         | 4000/5000 [06:00<01:30] Actor Loss: 0.3456 | Value Loss: 0.0989 | GPD Loss: 0.5321 | Tail Events: 100,123
Training:  90%|██████████████████████▌      | 4500/5000 [06:45<00:45] Actor Loss: 0.2345 | Value Loss: 0.0654 | GPD Loss: 0.5489 | Tail Events: 112,345
Training: 100%|████████████████████████████| 5000/5000 [07:30<00:00] Actor Loss: 0.1245 | Value Loss: 0.0321 | GPD Loss: 0.5612 | Tail Events: 125,680

================================================================================
TRAINING COMPLETE ✓
================================================================================
Total updates: 5000
Time elapsed: 2.08 hours
Model saved: ./runs/training_run_01/checkpoint_final.pth
Log saved: ./runs/training_run_01/training_log.csv
================================================================================
```

---

## Features

✅ **Clean terminal** — Only progress bar + metrics  
✅ **Live update** — See % completion in real-time  
✅ **Key metrics** — Actor/Value/GPD losses + tail events  
✅ **10% checkpoints** — Metrics every 500 updates (for 5000 total)  
✅ **Detailed final summary** — Training time, model location, log location  
✅ **Verbose mode** — Set `trainer.verbose = True` to see debug output if needed  

---

## FAQ

**Q: Why only 10% intervals?**
A: Keeps the display clean. Full metrics at every update = terminal spam.

**Q: Can I show metrics at different intervals?**
A: Yes! Edit trainer_a2c.py line 486:
```python
# Change from total_updates // 10 to something else:
if progress_pct == 1.0 or (update + 1) % max(1, total_updates // 5) == 0:  # Every 20%
```

**Q: What if I need to see debug output during training?**
A: Set verbose mode before training:
```python
trainer.verbose = True
train_loop(trainer, ...)
```

**Q: How do I know if training is working?**
A: Watch these trends:
- Actor Loss: Should decrease (policy improving)
- Value Loss: Should decrease (estimates improving)  
- GPD Loss: Should increase or stay > 0 (tail risk detected)
- Tail Events: Should grow steadily

---

## Installation

If `tqdm` is not installed:
```bash
poetry add tqdm
```

Or:
```bash
pip install tqdm
```

It's already part of most PyTorch installations, so likely already available!
