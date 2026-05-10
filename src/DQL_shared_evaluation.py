import os
import random
import pickle
import torch
import torch.nn as nn
import numpy as np
import traci

# =====================
# CONFIG
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SUMO_BINARY = "sumo"
SUMO_CFG = os.path.join(BASE_DIR, "sumo", "simulation_2.sumocfg")

MODEL_FILE = os.path.join(RESULTS_DIR, "dqn_shared_fixed.pt")

TLS_IDS = ["J1", "J5"]

AGENTS = {
    "J1": {"NS": ["E2_0", "E3_0"], "EW": ["E1_0", "E0_0"]},
    "J5": {"NS": ["E5_0", "E7_0"], "EW": ["E6_0", "-E4_0"]}
}

OUT_EDGES = {
    "J1": ["-E0", "-E1", "-E2", "-E3"],
    "J5": ["E4", "-E5", "-E6", "-E7"]
}

PHASE_NS       = 0
PHASE_NS_YELLOW = 1
PHASE_EW       = 2
PHASE_EW_YELLOW = 3
ACTIONS = [PHASE_NS, PHASE_EW]

YELLOW_PHASE = {
    PHASE_NS: PHASE_NS_YELLOW,
    PHASE_EW: PHASE_EW_YELLOW,
}

T_DECISION     = 10
YELLOW_DURATION = 3   # must match training script
MAX_STEPS      = 600
EPISODES       = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# MODEL — must match training architecture exactly
# Input size is now 4: [ns_norm, ew_norm, onehot_0, onehot_1]
# =====================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),   # ✅ updated from 3 → 4
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# LOAD MODEL
# =====================

policy = DQN()
policy.load_state_dict(torch.load(MODEL_FILE))
policy.eval()

print("✅ DQN shared model loaded")

# =====================
# STATE — one-hot agent ID, matching training script exactly
# =====================

AGENT_ONEHOT = {
    "J1": [1.0, 0.0],
    "J5": [0.0, 1.0],
}

def get_state(tls_id):
    ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls_id]["NS"])
    ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls_id]["EW"])

    one_hot = AGENT_ONEHOT[tls_id]  # ✅ one-hot instead of scalar 0/1

    return np.array([ns / 10, ew / 10] + one_hot, dtype=np.float32)

# =====================
# METRICS
# =====================

def collect_metrics():
    total_w, total_q, total_f = 0, 0, 0

    for tls in TLS_IDS:
        for lane in AGENTS[tls]["NS"] + AGENTS[tls]["EW"]:
            total_w += traci.lane.getWaitingTime(lane)
            total_q += traci.lane.getLastStepHaltingNumber(lane)

        total_f += sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES[tls])

    return total_w, total_q, total_f

# =====================
# POLICY (greedy — no exploration)
# =====================

def choose_action(state):
    with torch.no_grad():
        s = torch.tensor(state).unsqueeze(0)
        return torch.argmax(policy(s)).item()

# =====================
# EPISODE
# =====================

def run_episode():

    traci.start([SUMO_BINARY, "-c", SUMO_CFG,
                 "--seed", str(random.randint(0, 1000000))])

    for tls in TLS_IDS:
        traci.trafficlight.setPhase(tls, PHASE_NS)

    step = 0
    total_w, total_q, total_f = 0, 0, 0

    # ✅ Yellow phase tracking — mirrors training script exactly
    current_green   = {tls: PHASE_NS for tls in TLS_IDS}
    yellow_countdown = {tls: 0        for tls in TLS_IDS}
    pending_green   = {tls: None      for tls in TLS_IDS}

    while step < MAX_STEPS:

        traci.simulationStep()
        step += 1

        # Advance yellow transitions
        for tls in TLS_IDS:
            if yellow_countdown[tls] > 0:
                yellow_countdown[tls] -= 1
                if yellow_countdown[tls] == 0 and pending_green[tls] is not None:
                    traci.trafficlight.setPhase(tls, pending_green[tls])
                    current_green[tls] = pending_green[tls]
                    pending_green[tls] = None

        # Collect metrics every step
        w, q, f = collect_metrics()
        total_w += w
        total_q += q
        total_f += f

        # Decision point
        if step % T_DECISION == 0:
            for tls in TLS_IDS:
                state        = get_state(tls)
                action       = choose_action(state)
                desired_green = ACTIONS[action]

                if desired_green != current_green[tls]:
                    # Switch: go through yellow first
                    yellow_phase = YELLOW_PHASE[current_green[tls]]
                    traci.trafficlight.setPhase(tls, yellow_phase)
                    pending_green[tls]    = desired_green
                    yellow_countdown[tls] = YELLOW_DURATION
                else:
                    # No switch: re-affirm current green
                    traci.trafficlight.setPhase(tls, desired_green)

    traci.close()

    avg_w = total_w / MAX_STEPS
    avg_q = total_q / MAX_STEPS
    avg_f = total_f / MAX_STEPS

    return avg_w, avg_q, avg_f

# =====================
# MAIN
# =====================

if __name__ == "__main__":

    waits, queues, flows = [], [], []

    for ep in range(EPISODES):

        w, q, f = run_episode()

        waits.append(w)
        queues.append(q)
        flows.append(f)

        print(
            f"DQN | Ep {ep+1}/{EPISODES} | "
            f"W: {w:.2f} | Q: {q:.2f} | F: {f:.2f}"
        )

    # =====================
    # FINAL RESULTS
    # =====================

    avg_waiting = sum(waits) / EPISODES
    avg_queue   = sum(queues) / EPISODES
    avg_flow    = sum(flows) / EPISODES

    print("\n===== DQN SHARED RESULTS =====")
    print(f"Avg Waiting : {avg_waiting:.2f}")
    print(f"Avg Queue   : {avg_queue:.2f}")
    print(f"Avg Flow    : {avg_flow:.2f}")

    # =====================
    # SAVE RESULTS
    # =====================

    results = {
        "waiting_times": waits,
        "queue_lengths" : queues,
        "flows"         : flows,
        "avg_waiting"   : avg_waiting,
        "avg_queue"     : avg_queue,
        "avg_flow"      : avg_flow
    }

    file_path = os.path.join(RESULTS_DIR, f"DQN_shared_eval_{EPISODES}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    print("💾 DQN evaluation results saved.")