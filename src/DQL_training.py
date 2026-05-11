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

EPISODES = 5000
MAX_STEPS = 600
SAVE_EVERY = 10

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_FILE = os.path.join(RESULTS_DIR, "dqn_model.pt")

#os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# DQN PARAMETERS
# =====================

GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 50

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999

# Reward weights (same as QL)
w_queue = -1
w_waiting = -1
w_flow = 1

# =====================
# STATE (same logic as QL but continuous)
# =====================

def get_state():
    ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepHaltingNumber(e) for e in EW_EDGES)
    return np.array([ns / 10.0, ew / 10.0], dtype=np.float32)

# =====================
# METRICS
# =====================

def collect_metrics():
    waiting = sum(traci.edge.getWaitingTime(e) for e in NS_EDGES + EW_EDGES)
    queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES + EW_EDGES)
    flow = sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES)
    return waiting, queue, flow

# =====================
# REWARD (same as QL)
# =====================

def get_reward(waiting, queue, flow):
    return (
        w_waiting * (waiting / 10.0) +
        w_queue * queue +
        w_flow * flow
    )

# =====================
# DQN MODEL
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

    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--seed", str(random.randint(0, 1000000))])
    traci.trafficlight.setPhase(TLS_ID, PHASE_NS)

    state = get_state()
    step = 0
    last_switch = -MIN_GREEN
    decision_count = 0

    total_reward = 0
    total_waiting = 0
    total_queue = 0
    total_flow = 0

    # interval accumulators (same as QL)
    int_wait = int_queue = int_flow = 0

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        waiting, queue, flow = collect_metrics()

        total_waiting += waiting
        total_queue += queue
        total_flow += flow

        int_wait += waiting
        int_queue += queue
        int_flow += flow

        if step % T_DECISION == 0:

            action = choose_action(state, policy)
            phase = [PHASE_NS, PHASE_EW][action]

            if step - last_switch >= MIN_GREEN:
                traci.trafficlight.setPhase(TLS_ID, phase)
                last_switch = step

            avg_w = int_wait / T_DECISION
            avg_q = int_queue / T_DECISION
            avg_f = int_flow / T_DECISION

            reward = get_reward(avg_w, avg_q, avg_f)

            next_state = get_state()
            done = (step >= MAX_STEPS)

            memory.add(state, action, reward, next_state, done)
            train_step(policy, target, optimizer, memory)

            decision_count += 1
            if decision_count % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

            state = next_state
            total_reward += reward

            int_wait = int_queue = int_flow = 0

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

        print(f"Ep {ep+1} | R: {r:.2f} | W: {w:.2f} | Q: {q:.2f} | F: {f:.2f} | eps: {epsilon:.3f}")

        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(policy.state_dict(), MODEL_FILE)
            print("💾 Model saved")

    # save results
    data = {
        "rewards": rewards,
        "waiting_times": waits,
        "queue_lengths": queues,
        "flows": flows
    }

    with open(os.path.join(RESULTS_DIR, f"DQN_results{EPISODES}.pkl"), "wb") as f:
        pickle.dump(data, f)

    torch.save(policy.state_dict(), MODEL_FILE)

    print("🎉 DQN training complete")