#!/usr/bin/env python
"""Quick smoke test to verify all fixes."""
from run_smoke import trainer, train_loop

print("Testing with minimal params: 2 envs, 5 rollout steps, 1 update...")
trainer.num_envs = 2
trainer.rollout_length = 5
train_loop(trainer, total_updates=1, eval_interval=1, save_interval=1, csv_path='./runs/quick_test.csv')
print("\n✓✓✓ SUCCESS: Full training loop completed without errors! ✓✓✓")
