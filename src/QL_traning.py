import traci
import random
import pickle
import os

# =====================
# CONFIG
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_BINARY = "sumo"  # use "sumo-gui" for visualization
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
Q_TABLE_FILE = os.path.join(RESULTS_DIR, "q_table_2.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# Q-LEARNING PARAMETERS
# =====================

alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999


# Reward weights
w_queue = -1
w_waiting = -1
w_flow = 1



# =====================
# STATE DISCRETIZATION
# =====================

BINS = [0, 3, 6, 9, 12]

def to_bin(x):
    for i, b in enumerate(BINS):
        if x <= b:
            return i
    return len(BINS)

def get_state():
    ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepHaltingNumber(e) for e in EW_EDGES)
    return (to_bin(ns), to_bin(ew))

def get_reward(waiting,queue,flow):
    waiting_norm = waiting / 10
    queue_norm = queue / 1
    flow_norm = flow / 1

    return (
        w_waiting * waiting_norm +
        w_queue * queue_norm +
        w_flow * flow_norm
    )

def collect_metrics():
    waiting = sum(traci.edge.getWaitingTime(e) for e in NS_EDGES + EW_EDGES)
    queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES + EW_EDGES)
    step_passed = sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES)
    #arrived = traci.simulation.getArrivedNumber()
    return waiting, queue, step_passed

# =====================
# LOAD / INIT Q-TABLE
# =====================


Q = {}
print("🆕 New Q-table created.")

ACTIONS = [PHASE_NS, PHASE_EW]

# =====================
# Q FUNCTIONS
# =====================

def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, len(ACTIONS) - 1)

    values = [Q.get((state, a), 0.0) for a in range(len(ACTIONS))]
    return int(max(range(len(ACTIONS)), key=lambda i: values[i]))

def update_q(state, action, reward, next_state):
    key = (state, action)
    qsa = Q.get(key, 0.0)

    next_vals = [Q.get((next_state, a), 0.0) for a in range(len(ACTIONS))]
    Q[key] = qsa + alpha * (reward + gamma * max(next_vals) - qsa)

# =====================
# TRAIN EPISODE
# =====================

def run_episode():
    global epsilon

    total_reward = 0
    total_waiting = 0
    total_queue = 0
    total_throughput = 0
    total_passed = 0
    
    interval_waiting = 0
    interval_queue = 0
    interval_passed = 0


    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--seed", str(random.randint(0, 1000000))])

    step = 0
    last_switch_step = -MIN_GREEN

    state = get_state()
    traci.trafficlight.setPhase(TLS_ID, PHASE_NS)

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        waiting, queue, step_passed = collect_metrics()

        total_waiting += waiting
        total_queue += queue
        total_passed += step_passed
        
        interval_waiting += waiting
        interval_queue += queue
        interval_passed += step_passed
        
       

        # =====================
        # ACTION STEP
        # =====================
        if step % T_DECISION == 0:

            action_idx = choose_action(state)
            chosen_phase = ACTIONS[action_idx]

            # enforce safety constraint
            if step - last_switch_step >= MIN_GREEN:
                traci.trafficlight.setPhase(TLS_ID, chosen_phase)
                last_switch_step = step
                
            avg_interval_waiting = interval_waiting/T_DECISION
            avg_interval_queue = interval_queue/T_DECISION
            avg_interval_flow = interval_passed/T_DECISION
        

            next_state = get_state()
            reward = get_reward(avg_interval_waiting,avg_interval_queue,avg_interval_flow)
            total_reward += reward
            
            interval_waiting = 0
            interval_queue = 0
            interval_passed = 0

            update_q(state, action_idx, reward, next_state)
            state = next_state

    traci.close()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    avg_waiting = total_waiting / MAX_STEPS
    avg_queue = total_queue / MAX_STEPS
    avg_flow = total_passed / MAX_STEPS

    return total_reward, avg_waiting, avg_queue, avg_flow

# =====================
# TRAIN LOOP
# =====================

if __name__ == "__main__":

    rewards = []
    waiting_times = []
    queue_lengths = []
    flows = []

    for ep in range(EPISODES):

        r, w, q, f = run_episode()

        rewards.append(r)
        waiting_times.append(w)
        queue_lengths.append(q)
        flows.append(f)

        print(
            f"Episode {ep+1}/{EPISODES} | "
            f"Reward: {r:.2f} | "
            f"Wait: {w:.2f} | "
            f"Queue: {q:.2f} | "
            f"Flow: {f:.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )

        if (ep + 1) % SAVE_EVERY == 0:
            with open(Q_TABLE_FILE, "wb") as f:
                pickle.dump(Q, f)
            print(f"💾 Q-table saved at episode {ep+1}")

    # =====================
    # SAVE TRAINING DATA
    # =====================

    training_data = {
        "rewards": rewards,
        "waiting_times": waiting_times,
        "queue_lengths": queue_lengths,
        "flows": flows
    }

    file_path = os.path.join(RESULTS_DIR, f"QL_trainingData_betterReward{EPISODES}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(training_data, f)

    # final save
    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(Q, f)

    print("🎉 Training complete. Everything saved.")

