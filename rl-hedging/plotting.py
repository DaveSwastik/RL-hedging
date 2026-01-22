import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
FILE_PATH = "training_data_log.xlsx"   # training log output from train_era_rl.py
SHEET_NAME = "Sheet1"                 # Excel sheet name
PRICE_COL = "Price"       # Column D
ACTION_COL = "Action"     # Column F
TIME_COL = None           # set to "Step" if you want explicit time

DOWNSAMPLE_EVERY = 10     # increase if data is huge (e.g. 50, 100)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

# If no explicit time column, use index as time
if TIME_COL and TIME_COL in df.columns:
    t = df[TIME_COL]
else:
    # df.index is a RangeIndex; convert to Series for consistent iloc slicing
    t = pd.Series(df.index, name="Index")

price = df[PRICE_COL]
action = df[ACTION_COL]

# -----------------------------
# DOWNSAMPLE (important for speed)
# -----------------------------
t_ds = t.iloc[::DOWNSAMPLE_EVERY]
price_ds = price.iloc[::DOWNSAMPLE_EVERY]
action_ds = action.iloc[::DOWNSAMPLE_EVERY]

# -----------------------------
# PLOT
# -----------------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(14, 8),
    gridspec_kw={"height_ratios": [2, 1]}
)

# --- Price ---
axes[0].plot(t_ds, price_ds)
axes[0].set_ylabel("Price")
axes[0].set_title("Price and Hedge Ratio over Time")
axes[0].grid(True, alpha=0.3)

# --- Hedge Ratio (Action) ---
axes[1].plot(t_ds, action_ds)
axes[1].set_ylabel("Hedge Ratio (Action)")
axes[1].set_xlabel("Time")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
