import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =====================
# Load file
# =====================

file_name = input("Data file name (without .pkl): ") + ".pkl"
file_path = os.path.join(RESULTS_DIR, file_name)

with open(file_path, "rb") as f:
    data = pickle.load(f)

# =====================
# Extract metrics
# =====================

waiting_times = np.array(data.get("waiting_times", []))
queue_lengths = np.array(data.get("queue_lengths", []))
flows = np.array(data.get("flows", []))

# =====================
# Compute averages
# =====================

avg_wait = np.mean(waiting_times) if len(waiting_times) > 0 else 0
avg_queue = np.mean(queue_lengths) if len(queue_lengths) > 0 else 0
avg_flow = np.mean(flows) if len(flows) > 0 else 0

print("\n===== AVERAGES =====")
print(f"Waiting Time: {avg_wait:.2f}")
print(f"Queue Length: {avg_queue:.2f}")
print(f"Flow: {avg_flow:.2f}")

# =====================
# BAR CHART
# =====================

metrics = ["Waiting Time", "Queue Length", "Flow"]
values = [avg_wait, avg_queue, avg_flow]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values)

plt.title("Average Performance Metrics")
plt.ylabel("Value")
plt.grid(axis='y')

plt.tight_layout()
plt.show()