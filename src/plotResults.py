import matplotlib.pyplot as plt
import pickle
import numpy as np



# Load rewards

TrainingDataFileName = input(" training data file name (without the .pkl) : ") + ".pkl"

#if not os.path.exists(TrainingDataFileName):
#    print("❌ No trained Q-table found.")
#    exit()


with open(TrainingDataFileName, "rb") as f:
    rewards = pickle.load(f)

rewards = np.array(rewards)

# ---- Moving Average ----
window_size = len(rewards)//20
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# ---- Plot ----
plt.figure()

plt.plot(rewards, alpha=0.3, label="Raw Reward")
plt.plot(range(window_size-1, len(rewards)), moving_avg, linewidth=2, label="Moving Average")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Curve")
plt.legend()
plt.grid(True)

plt.show()
