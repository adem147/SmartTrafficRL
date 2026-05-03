import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =====================
# Load file
# =====================

TrainingDataFileName = input("Training data file name (without the .pkl): ") + ".pkl"
TrainingDataFileName = os.path.join(RESULTS_DIR, TrainingDataFileName)

with open(TrainingDataFileName, "rb") as f:
    data = pickle.load(f)

# =====================
# Handle BOTH formats
# =====================

if isinstance(data, dict):
    rewards = np.array(data.get("rewards", []))
    waiting_times = np.array(data.get("waiting_times", []))
    queue_lengths = np.array(data.get("queue_lengths", []))
    throughputs = np.array(data.get("flows", []))
else:
    rewards = np.array(data)
    waiting_times = np.array([])
    queue_lengths = np.array([])
    throughputs = np.array([])

# =====================
# ASYMPTOTIC VALUE
# =====================

def get_asymptotic_value(values, ratio=0.1):
    if values is None or len(values) == 0:
        return None
    
    n = len(values)
    k = max(1, int(n * ratio))  # last 10%
    
    return np.mean(values[-k:])

# =====================
# Plot function
# =====================

def plot_metric(ax, values, title, ylabel):
    if values is None or len(values) == 0:
        ax.set_title(title + " (NO DATA)")
        return

    values = np.array(values)

    window_size = max(1, len(values) // 20)

    if len(values) < window_size:
        moving_avg = values
        x_ma = range(len(values))
    else:
        moving_avg = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
        x_ma = range(window_size - 1, len(values))

    # asymptotic value
    asym_val = get_asymptotic_value(values)

    # plots
    ax.plot(values, alpha=0.3, label="Raw")
    ax.plot(x_ma, moving_avg, linewidth=2, label="Moving Avg")

    # horizontal asymptote
    if asym_val is not None:
        ax.axhline(asym_val, linestyle='--', linewidth=2, label=f"Asymptote: {asym_val:.2f}")

        # optional: highlight convergence zone
        start = int(len(values) * 0.9)
        ax.axvspan(start, len(values), alpha=0.1)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

# =====================
# PRINT SUMMARY
# =====================

episodes = len(rewards)

print(f"avg reward: {(sum(rewards)/episodes):.2f}")
print(f"avg waiting time: {(sum(waiting_times)/episodes):.2f}")
print(f"avg queue length: {(sum(queue_lengths)/episodes):.2f}")

print("\n--- Asymptotic values (last 10%) ---")
print(f"asymptotic reward: {get_asymptotic_value(rewards):.2f}")
print(f"asymptotic waiting time: {get_asymptotic_value(waiting_times):.2f}")
print(f"asymptotic queue length: {get_asymptotic_value(queue_lengths):.2f}")
print(f"asymptotic flow: {get_asymptotic_value(throughputs):.2f}")

# =====================
# PLOTS
# =====================

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

plot_metric(axs[0, 0], rewards, "Reward per Episode", "Reward")
plot_metric(axs[0, 1], waiting_times, "Waiting Time per Episode", "Waiting Time")
plot_metric(axs[1, 0], queue_lengths, "Queue Length per Episode", "Queue Length")
plot_metric(axs[1, 1], throughputs, "Flow per Episode", "Vehicles")

plt.tight_layout()
plt.show()