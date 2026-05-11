import traci
import random
import pickle
import os

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_BINARY = "sumo"
SUMO_CFG = os.path.join(BASE_DIR, "sumo", "simulation_2.sumocfg")

TLS_IDS = ["J1", "J5"]

AGENTS = {
    "J1": {
        "NS": ["E2_0", "E3_0"],
        "EW": ["E1_0", "E0_0"]
    },
    "J5": {
        "NS": ["E5_0", "E7_0"],
        "EW": ["E6_0", "-E4_0"]
    }
}

OUT_EDGES = {
    "J1": ["-E0", "-E1", "-E2", "-E3"],
    "J5": ["E4", "-E5", "-E6", "-E7"]
}

PHASE_NS = 0
PHASE_EW = 2
ACTIONS = [PHASE_NS, PHASE_EW]

# =========================
# HYPERPARAMETERS
# =========================

alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999

T_DECISION = 10
MAX_STEPS = 600
EPISODES = 5000
SAVE_EVERY = 10

# reward weights
w_waiting = -1
w_queue = -1
w_flow = 1
w_global = 0.3

# =========================
# FILES
# =========================

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

Q_TABLE_FILE = os.path.join(RESULTS_DIR, "q_table_IQL.pkl")

# =========================
# DISCRETIZATION
# =========================

BINS = [0, 3, 6, 9, 12]

def to_bin(x):
    for i, b in enumerate(BINS):
        if x <= b:
            return i
    return len(BINS)

# =========================
# Q TABLE
# =========================


Q = {tls: {} for tls in TLS_IDS}
print("🆕 New Q-table created")

# =========================
# STATE
# =========================

def get_state(tls_id):
    ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls_id]["NS"])
    ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls_id]["EW"])
    return (to_bin(ns), to_bin(ew))

# =========================
# LOCAL METRICS
# =========================

def collect_local_metrics(tls_id):
    waiting = 0
    queue = 0
    flow = 0

    for lane in AGENTS[tls_id]["NS"] + AGENTS[tls_id]["EW"]:
        waiting += traci.lane.getWaitingTime(lane)
        queue += traci.lane.getLastStepHaltingNumber(lane)
    
    flow = sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES[tls_id])

    return waiting, queue, flow

# =========================
# GLOBAL METRICS
# =========================

def collect_global_metrics():
    total_w, total_q, total_f = 0, 0, 0

    for tls in TLS_IDS:
        w, q, f = collect_local_metrics(tls)
        total_w += w
        total_q += q
        total_f += f

    return total_w, total_q, total_f

# =========================
# REWARD (CLEAN)
# =========================

def get_reward(waiting, queue, flow):
    waiting_norm = waiting / 10
    queue_norm = queue / 1
    flow_norm = flow / 1   
    
    return (
       w_waiting * waiting_norm +
        w_queue * queue_norm +
        w_flow * flow_norm
    )

# =========================
# RL FUNCTIONS
# =========================

def choose_action(state, tls_id):
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))

    values = [Q[tls_id].get((state, a), 0.0) for a in range(len(ACTIONS))]
    return int(max(range(len(ACTIONS)), key=lambda i: values[i]))

def update_q(state, action, reward, next_state, tls_id):
    key = (state, action)

    qsa = Q[tls_id].get(key, 0.0)
    next_vals = [Q[tls_id].get((next_state, a), 0.0) for a in range(len(ACTIONS))]

    Q[tls_id][key] = qsa + alpha * (reward + gamma * max(next_vals) - qsa)

# =========================
# EPISODE
# =========================

def run_episode():
    global epsilon

    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--seed", str(random.randint(0, 1000000))])

    step = 0
    total_reward = 0

    total_waiting = 0
    total_queue = 0
    total_flow = 0

    # interval accumulators per agent
    
    interval_w = {tls: 0 for tls in TLS_IDS}
    interval_q = {tls: 0 for tls in TLS_IDS}
    interval_f = {tls: 0 for tls in TLS_IDS}

    # initialize lights
    
    for tls in TLS_IDS:
        traci.trafficlight.setPhase(tls, PHASE_NS)

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        
        gw, gq, gf = collect_global_metrics()
        total_waiting += gw
        total_queue += gq
        total_flow += gf
        
        
        
        for tls in TLS_IDS:
            w, q, f = collect_local_metrics(tls)
            interval_w[tls] += w
            interval_q[tls] += q
            interval_f[tls] += f
            
            
        
       
        if step % T_DECISION == 0:
            
            local_rewards = {}

            for tls in TLS_IDS:
                avg_w = interval_w[tls] / T_DECISION
                avg_q = interval_q[tls] / T_DECISION
                avg_f = interval_f[tls] / T_DECISION

                local_rewards[tls] = get_reward(avg_w, avg_q, avg_f)

            global_reward = sum(local_rewards.values())
            
            for tls in TLS_IDS:

                state = get_state(tls)
                action = choose_action(state, tls)                
               
                reward = local_rewards[tls] + w_global*(global_reward-local_rewards[tls]) #take in consederation the global reward to get better cooperation 
                

                next_state = get_state(tls)

                update_q(state, action, reward, next_state, tls)

                traci.trafficlight.setPhase(tls, ACTIONS[action])

                total_reward += reward

               
            for tls in TLS_IDS:
                interval_w[tls] = 0
                interval_q[tls] = 0
                interval_f[tls] = 0
                

    traci.close()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return (
        total_reward,
        total_waiting / MAX_STEPS,
        total_queue / MAX_STEPS,
        total_flow / MAX_STEPS
    )

# =========================
# TRAINING LOOP
# =========================

if __name__ == "__main__":

    rewards, waits, queues, flows = [], [], [], []

    for ep in range(EPISODES):

        r, w, q, f = run_episode()

        rewards.append(r)
        waits.append(w)
        queues.append(q)
        flows.append(f)

        print(
            f"Ep {ep+1}/{EPISODES} | "
            f"R: {r:.2f} | W: {w:.2f} | Q: {q:.2f} | F: {f:.2f} | eps: {epsilon:.3f}"
        )

        if (ep + 1) % SAVE_EVERY == 0:
            with open(Q_TABLE_FILE, "wb") as f:
                pickle.dump(Q, f)
            print("💾 Q-table saved")

    # save results
    data = {
        "rewards": rewards,
        "waiting_times": waits,
        "queue_lengths": queues,
        "flows": flows
    }

    with open(os.path.join(RESULTS_DIR, f"IQL_results_betterReward{EPISODES}.pkl"), "wb") as f:
        pickle.dump(data, f)

    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(Q, f)

    print("🎉 Training complete")