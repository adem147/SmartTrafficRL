import traci
import pickle
import time
import os
import random

# ===============================
# CONFIG
# ===============================

SUMO_BINARY = "sumo-gui"   # GUI version
SUMO_CFG = "simulation.sumocfg"

NS_EDGES = ["-E2", "-E3"]
EW_EDGES = ["-E1", "E0"]

PHASE_NS = 0
PHASE_EW = 2
TLS_ID = "J1"

T_DECISION = 10
MIN_GREEN = 5
MAX_STEPS = 600

Q_TABLE_FILE = "q_table.pkl"

# ===============================
# LOAD Q-TABLE
# ===============================

if not os.path.exists(Q_TABLE_FILE):
    print("❌ No trained Q-table found.")
    exit()

with open(Q_TABLE_FILE, "rb") as f:
    Q = pickle.load(f)

print("✅ Q-table loaded. Starting visualization...")

ACTIONS = [PHASE_NS, PHASE_EW]

# ===============================
# STATE DISCRETIZATION
# ===============================

BINS = [0, 3, 7]

def to_bin(x):
    for i, b in enumerate(BINS):
        if x <= b:
            return i
    return len(BINS)

def get_state():
    ns = sum(traci.edge.getLastStepVehicleNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepVehicleNumber(e) for e in EW_EDGES)
    return (to_bin(ns), to_bin(ew))

def get_reward():
    waiting =  (sum(traci.edge.getWaitingTime(e) for e in NS_EDGES) * 1.0 +
            sum(traci.edge.getWaitingTime(e) for e in EW_EDGES) * 1.0)
    return - (waiting)

# ===============================
# GREEDY POLICY (epsilon = 0)
# ===============================

def choose_best_action(state):
    values = [Q.get((state, a), 0.0) for a in range(len(ACTIONS))]
    return int(max(range(len(ACTIONS)), key=lambda i: values[i]))

# ===============================
# VISUAL RUN
# ===============================

def run_visual_episode():

    sumo_seed = random.randint(0, 1000000)
    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--seed", str(sumo_seed)])


    step = 0
    last_switch_step = -MIN_GREEN
    total_reward = 0

    traci.trafficlight.setPhase(TLS_ID, PHASE_NS)

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        time.sleep(0.05)  # slow down so you can see it

        if step % T_DECISION == 0:
            state = get_state()
            action_idx = choose_best_action(state)
            chosen_phase = ACTIONS[action_idx]

            if (step - last_switch_step >= MIN_GREEN and
                traci.trafficlight.getPhase(TLS_ID) != chosen_phase):
                traci.trafficlight.setPhase(TLS_ID, chosen_phase)
                last_switch_step = step

            reward = get_reward()
            total_reward += reward

            print(f"Step {step} | State {state} | Action {action_idx} | Reward {reward:.2f}")

    print(f"\n🎯 Total Episode Reward: {total_reward:.2f}")
    traci.close()


if __name__ == "__main__":
    run_visual_episode()
