import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import traci

# =====================
# CONFIG
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMO_BINARY = "sumo"
SUMO_CFG = os.path.join(BASE_DIR, "sumo", "simulation_2.sumocfg")
SAVE_EVERY = 100
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

PHASE_NS = 0
PHASE_NS_YELLOW = 1   # yellow after NS green
PHASE_EW = 2
PHASE_EW_YELLOW = 3   # yellow after EW green
ACTIONS = [PHASE_NS, PHASE_EW]

# Map each green phase to its corresponding yellow phase
YELLOW_PHASE = {
    PHASE_NS: PHASE_NS_YELLOW,
    PHASE_EW: PHASE_EW_YELLOW,
}

T_DECISION = 10
YELLOW_DURATION = 2   # sim steps for yellow before switching to new green
MAX_STEPS = 600
EPISODES = 3000

# =====================
# DQN PARAMS
# =====================

GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 50

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999

# reward weights
w_waiting = -1
w_queue = -1
w_flow = 1

# =====================
# STATE
# State: [ns_queue_norm, ew_queue_norm, agent_id_onehot_0, agent_id_onehot_1]
# One-hot encoding avoids the implicit ordinal relationship of scalar 0/1 IDs.
# =====================

# Fixed one-hot vectors per agent
AGENT_ONEHOT = {
    "J1": [1.0, 0.0],
    "J5": [0.0, 1.0],
}

def get_local_state(tls_id):
    ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls_id]["NS"])
    ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls_id]["EW"])

    one_hot = AGENT_ONEHOT[tls_id]  # [1,0] or [0,1] — no ordinal bias

    return np.array([ns / 10, ew / 10] + one_hot, dtype=np.float32)

# =====================
# METRICS
# =====================

def collect_metrics(tls_id):
    waiting = 0
    queue = 0

    for lane in AGENTS[tls_id]["NS"] + AGENTS[tls_id]["EW"]:
        waiting += traci.lane.getWaitingTime(lane)
        queue += traci.lane.getLastStepHaltingNumber(lane)

    flow = sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES[tls_id])

    return waiting, queue, flow

# =====================
# REWARD
# =====================

def get_reward(waiting, queue, flow):
    return (
        w_waiting * (waiting / 10.0) +
        w_queue * queue +
        w_flow * flow
    )

# =====================
# MODEL
# Input size is now 4: [ns_norm, ew_norm, onehot_0, onehot_1]
# =====================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),   # 4 inputs: ns, ew, agent_onehot[0], agent_onehot[1]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# REPLAY BUFFER
# =====================

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_SIZE)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)

        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# =====================
# ACTION
# =====================

def choose_action(state, net):
    global epsilon

    if random.random() < epsilon:
        return random.randint(0, 1)

    with torch.no_grad():
        s = torch.tensor(state).unsqueeze(0)
        return torch.argmax(net(s)).item()

# =====================
# TRAIN STEP
# =====================

def train_step(policy, target, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return

    s, a, r, s2, d = memory.sample()

    q = policy(s).gather(1, a.unsqueeze(1)).squeeze()
    q_next = target(s2).max(1)[0]

    target_q = r + GAMMA * q_next * (1 - d)

    loss = nn.MSELoss()(q, target_q.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =====================
# EPISODE
# =====================

def run_episode(policy, target, optimizer, memory):
    global epsilon

    traci.start([SUMO_BINARY, "-c", SUMO_CFG,
                 "--seed", str(random.randint(0, 1000000))])

    for tls in TLS_IDS:
        traci.trafficlight.setPhase(tls, PHASE_NS)

    step = 0
    states = {tls: get_local_state(tls) for tls in TLS_IDS}
    decision_counter = 0

    total_reward = 0
    total_waiting = 0
    total_queue = 0
    total_flow = 0

    interval_w = {tls: 0 for tls in TLS_IDS}
    interval_q = {tls: 0 for tls in TLS_IDS}
    interval_f = {tls: 0 for tls in TLS_IDS}

    # --- Yellow phase transition tracking ---
    # current_green: the green phase each TLS is on (or heading to)
    current_green = {tls: PHASE_NS for tls in TLS_IDS}
    # yellow_countdown: steps remaining in yellow; 0 means not in yellow
    yellow_countdown = {tls: 0 for tls in TLS_IDS}
    # pending_green: the green phase to apply once yellow expires
    pending_green = {tls: None for tls in TLS_IDS}

    while step < MAX_STEPS:

        traci.simulationStep()
        step += 1

        # --- Advance yellow transitions ---
        for tls in TLS_IDS:
            if yellow_countdown[tls] > 0:
                yellow_countdown[tls] -= 1
                if yellow_countdown[tls] == 0 and pending_green[tls] is not None:
                    traci.trafficlight.setPhase(tls, pending_green[tls])
                    current_green[tls] = pending_green[tls]
                    pending_green[tls] = None

        # --- Collect metrics every step ---
        for tls in TLS_IDS:
            w, q, f = collect_metrics(tls)

            total_waiting += w
            total_queue += q
            total_flow += f

            interval_w[tls] += w
            interval_q[tls] += q
            interval_f[tls] += f

        # --- Decision point ---
        if step % T_DECISION == 0:

            actions = {}

            for tls in TLS_IDS:
                a = choose_action(states[tls], policy)
                actions[tls] = a
                desired_green = ACTIONS[a]

                if desired_green != current_green[tls]:
                    # Switch needed: go through yellow first
                    yellow_phase = YELLOW_PHASE[current_green[tls]]
                    traci.trafficlight.setPhase(tls, yellow_phase)
                    pending_green[tls] = desired_green
                    yellow_countdown[tls] = YELLOW_DURATION
                else:
                    # No switch: re-affirm the current green (already set)
                    traci.trafficlight.setPhase(tls, desired_green)

            done = (step >= MAX_STEPS)

            # Store experiences and compute rewards
            for tls in TLS_IDS:
                avg_w = interval_w[tls] / T_DECISION
                avg_q = interval_q[tls] / T_DECISION
                avg_f = interval_f[tls] / T_DECISION

                reward = get_reward(avg_w, avg_q, avg_f)
                next_state = get_local_state(tls)

                memory.add(states[tls], actions[tls], reward, next_state, done)

                states[tls] = next_state
                total_reward += reward

                interval_w[tls] = 0
                interval_q[tls] = 0
                interval_f[tls] = 0

            # Train once per agent — matches the 2x experience throughput IQL gets
            for _ in TLS_IDS:
                train_step(policy, target, optimizer, memory)

            decision_counter += 1
            if decision_counter % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

    traci.close()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return (
        total_reward,
        total_waiting / MAX_STEPS,
        total_queue / MAX_STEPS,
        total_flow / MAX_STEPS
    )

# =====================
# MAIN
# =====================

if __name__ == "__main__":

    policy = DQN()
    target = DQN()
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    memory = ReplayBuffer()

    rewards, waits, queues, flows = [], [], [], []

    for ep in range(EPISODES):

        r, w, q, f = run_episode(policy, target, optimizer, memory)

        rewards.append(r)
        waits.append(w)
        queues.append(q)
        flows.append(f)

        print(
            f"Ep {ep+1}/{EPISODES} | "
            f"R: {r:.2f} | W: {w:.2f} | Q: {q:.2f} | F: {f:.2f} | eps: {epsilon:.3f}"
        )

        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(policy.state_dict(), MODEL_FILE)
            print("💾 Model saved")

    data = {
        "rewards": rewards,
        "waiting_times": waits,
        "queue_lengths": queues,
        "flows": flows
    }

    with open(os.path.join(RESULTS_DIR, f"DQN_shared_fixed_{EPISODES}.pkl"), "wb") as f:
        pickle.dump(data, f)

    torch.save(policy.state_dict(), MODEL_FILE)

    print("🎉 Training complete")

