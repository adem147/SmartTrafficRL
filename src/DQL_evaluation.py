import traci
import random
import pickle
import os
import torch
import torch.nn as nn
import numpy as np

# =====================
# CONFIG (same as training)
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_BINARY = "sumo"
SUMO_CFG = os.path.join(BASE_DIR, "sumo", "simulation.sumocfg")

NS_EDGES = ["-E2", "-E3"]
EW_EDGES = ["-E1", "E0"]
OUT_EDGES = ["E1", "E2", "E3", "-E0"]

PHASE_NS = 0
PHASE_EW = 2
TLS_ID = "J1"

T_DECISION = 10
MIN_GREEN = 5

EPISODES = 100
MAX_STEPS = 600

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_FILE = os.path.join(RESULTS_DIR, "dqn_model.pt")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# DQN MODEL (same as training)
# =====================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# load model
policy = DQN()
policy.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
policy.eval()

print("✅ Model loaded")

# =====================
# STATE (EXACT SAME AS TRAINING)
# =====================

def get_state():
    ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepHaltingNumber(e) for e in EW_EDGES)
    return np.array([ns / 10.0, ew / 10.0], dtype=np.float32)

# =====================
# METRICS (SAME)
# =====================

def collect_metrics():
    waiting = sum(traci.edge.getWaitingTime(e) for e in NS_EDGES + EW_EDGES)
    queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES + EW_EDGES)
    flow = sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES)
    return waiting, queue, flow

# =====================
# POLICY (GREEDY ONLY, NO EPSILON)
# =====================

def choose_action(state):
    with torch.no_grad():
        s = torch.tensor(state).unsqueeze(0)
        return torch.argmax(policy(s)).item()

# =====================
# RUN EPISODE
# =====================

def run_episode():

    total_waiting = 0
    total_queue = 0
    total_flow = 0

    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--seed", str(random.randint(0, 1000000))])

    step = 0
    last_switch = -MIN_GREEN

    state = get_state()
    traci.trafficlight.setPhase(TLS_ID, PHASE_NS)

    while step < MAX_STEPS:

        traci.simulationStep()
        step += 1

        waiting, queue, flow = collect_metrics()

        total_waiting += waiting
        total_queue += queue
        total_flow += flow

        if step % T_DECISION == 0:

            action = choose_action(state)
            phase = [PHASE_NS, PHASE_EW][action]

            if step - last_switch >= MIN_GREEN:
                traci.trafficlight.setPhase(TLS_ID, phase)
                last_switch = step

            state = get_state()

    traci.close()

    return (
        total_waiting / MAX_STEPS,
        total_queue / MAX_STEPS,
        total_flow / MAX_STEPS
    )

# =====================
# MAIN LOOP
# =====================

if __name__ == "__main__":

    waiting_times = []
    queue_lengths = []
    flows = []

    for ep in range(EPISODES):

        w, q, f = run_episode()

        waiting_times.append(w)
        queue_lengths.append(q)
        flows.append(f)

        print(
            f"Episode {ep+1}/{EPISODES} | "
            f"Wait: {w:.2f} | "
            f"Queue: {q:.2f} | "
            f"Flow: {f:.2f}"
        )

    # =====================
    # FINAL RESULTS
    # =====================

    avg_waiting = sum(waiting_times) / EPISODES
    avg_queue = sum(queue_lengths) / EPISODES
    avg_flow = sum(flows) / EPISODES

    print("\n===== DQN EVALUATION RESULTS =====")
    print(f"Avg Waiting Time: {avg_waiting:.2f}")
    print(f"Avg Queue Length: {avg_queue:.2f}")
    print(f"Avg Flow: {avg_flow:.2f}")

    # =====================
    # SAVE RESULTS
    # =====================

    data = {
        "waiting_times": waiting_times,
        "queue_lengths": queue_lengths,
        "flows": flows
    }

    with open(os.path.join(RESULTS_DIR, f"DQN_eval_results_{EPISODES}.pkl"), "wb") as f:
        pickle.dump(data, f)

    print("💾 Evaluation results saved")