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
MODEL_FILE = os.path.join(RESULTS_DIR, "dqn_shared_intersections.pt")

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
PHASE_EW = 2
ACTIONS = [PHASE_NS, PHASE_EW]

T_DECISION = 10
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
# =====================

def get_shared_state():
    j1_ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS["J1"]["NS"])
    j1_ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS["J1"]["EW"])

    j5_ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS["J5"]["NS"])
    j5_ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS["J5"]["EW"])

    return np.array([j1_ns/10, j1_ew/10, j5_ns/10, j5_ew/10], dtype=np.float32)

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
# =====================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
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
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
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
    state = get_shared_state()
    decision_counter = 0

    total_reward = 0
    total_waiting = 0
    total_queue = 0
    total_flow = 0

    interval_w = {tls: 0 for tls in TLS_IDS}
    interval_q = {tls: 0 for tls in TLS_IDS}
    interval_f = {tls: 0 for tls in TLS_IDS}

    while step < MAX_STEPS:

        traci.simulationStep()
        step += 1

        for tls in TLS_IDS:
            w, q, f = collect_metrics(tls)

            total_waiting += w
            total_queue += q
            total_flow += f

            interval_w[tls] += w
            interval_q[tls] += q
            interval_f[tls] += f

        if step % T_DECISION == 0:

            action = choose_action(state, policy)
            phase = ACTIONS[action]

            for tls in TLS_IDS:
                traci.trafficlight.setPhase(tls, phase)

            reward_total = 0

            for tls in TLS_IDS:
                avg_w = interval_w[tls] / T_DECISION
                avg_q = interval_q[tls] / T_DECISION
                avg_f = interval_f[tls] / T_DECISION

                reward_total += get_reward(avg_w, avg_q, avg_f)

                interval_w[tls] = 0
                interval_q[tls] = 0
                interval_f[tls] = 0

            next_state = get_shared_state()
            done = (step >= MAX_STEPS)

            memory.add(state, action, reward_total, next_state, done)
            train_step(policy, target, optimizer, memory)

            state = next_state
            total_reward += reward_total

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


    # save results (same format as IQL)
    data = {
        "rewards": rewards,
        "waiting_times": waits,
        "queue_lengths": queues,
        "flows": flows
    }

    with open(os.path.join(RESULTS_DIR, f"DQN_shared_results_{EPISODES}.pkl"), "wb") as f:
        pickle.dump(data, f)

    torch.save(policy.state_dict(), MODEL_FILE)

    print("🎉 Training complete")