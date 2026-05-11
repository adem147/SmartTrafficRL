import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# =====================
# LOAD DATA
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

file_name = input("Enter results file name (without .pkl): ") + ".pkl"
file_path = os.path.join(RESULTS_DIR, file_name)

with open(file_path, "rb") as f:
    data = pickle.load(f)

# =====================
# EXTRACT METRICS
# =====================

rewards = np.array(data.get("rewards", []))
waiting = np.array(data.get("waiting_times", []))
queues = np.array(data.get("queue_lengths", []))
flows = np.array(data.get("flows", []))

# =====================
# CORRELATION FUNCTION
# =====================

def corr(x, y):
    if len(x) == 0 or len(y) == 0:
        return None
    return np.corrcoef(x, y)[0, 1]

# =====================
# COMPUTE CORRELATIONS
# =====================

pairs = [
    ("Reward vs Waiting Time", rewards, waiting),
    ("Reward vs Queue Length", rewards, queues),
    ("Reward vs Flow", rewards, flows),
    ("Waiting Time vs Queue Length", waiting, queues),
    ("Waiting Time vs Flow", waiting, flows),
    ("Queue Length vs Flow", queues, flows),
]

print("\n📊 CORRELATION RESULTS")
print("=" * 40)

results = {}

for name, x, y in pairs:
    c = corr(x, y)
    results[name] = c
    print(f"{name}: {c:.4f}")

# =====================
# HEATMAP VISUALIZATION
# =====================

labels = ["Reward", "Waiting", "Queue", "Flow"]

data_matrix = np.array([
    rewards,
    waiting,
    queues,
    flows
])

corr_matrix = np.corrcoef(data_matrix)

plt.figure(figsize=(6, 5))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(range(4), labels)
plt.yticks(range(4), labels)

plt.title("Metric Correlation Heatmap")

# annotate values
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{corr_matrix[i, j]:.2f}",
                 ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

# =====================
# SCATTER PLOTS (KEY RELATIONS)
# =====================

def scatter(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, alpha=0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

scatter(waiting, queues, "Waiting vs Queue", "Queue", "Waiting")
scatter(queues, flows, "Queue vs Flow", "Flow", "Queue")
scatter(waiting, flows, "Waiting vs Flow", "Flow", "Waiting")

# =====================
# SUMMARY INSIGHT
# =====================

print("\n📌 KEY INSIGHT")
strong = [k for k, v in results.items() if abs(v) > 0.7]

print("Strong correlations (>0.7):")
for s in strong:
    print(" -", s)