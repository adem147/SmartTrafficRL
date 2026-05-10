import traci
import random
import pickle
import os

# =====================
# CONFIG
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_BINARY = "sumo"
SUMO_CFG = os.path.join(BASE_DIR, "sumo", "simulation_2.sumocfg")

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
MIN_GREEN = 5

EPISODES = 100
MAX_STEPS = 600

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

Q_TABLE_FILE = os.path.join(RESULTS_DIR, "q_table_IQL.pkl")

# =====================
# LOAD MODEL
# =====================

with open(Q_TABLE_FILE, "rb") as f:
    Q = pickle.load(f)

print("✅ IQL Q-table loaded")

# =====================
# DISCRETIZATION
# =====================

BINS = [0, 3, 6, 9, 12]

def to_bin(x):
    for i, b in enumerate(BINS):
        if x <= b:
            return i
    return len(BINS)

# =====================
# STATE
# =====================

def get_state(tls):
    ns = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls]["NS"])
    ew = sum(traci.lane.getLastStepHaltingNumber(l) for l in AGENTS[tls]["EW"])
    return (to_bin(ns), to_bin(ew))

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
# POLICY
# =====================

def choose_action(state, tls):
    values = [Q[tls].get((state, a), 0.0) for a in range(len(ACTIONS))]
    return int(max(range(len(ACTIONS)), key=lambda i: values[i]))

# =====================
# EPISODE
# =====================

def run_episode():

    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--seed", str(random.randint(0, 1000000))])

    step = 0
    total_w, total_q, total_f = 0, 0, 0

    last_switch = {tls: -MIN_GREEN for tls in TLS_IDS}

    for tls in TLS_IDS:
        traci.trafficlight.setPhase(tls, PHASE_NS)
        
    total_v = 0

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        w, q, f = collect_metrics()
        total_w += w
        total_q += q
        total_f += f

        if step % T_DECISION == 0:
            for tls in TLS_IDS:
                state = get_state(tls)
                action = choose_action(state, tls)

                # ✅ APPLY ACTION (FIX)
                if step - last_switch[tls] >= MIN_GREEN:
                    traci.trafficlight.setPhase(tls, ACTIONS[action])
                    last_switch[tls] = step
                    
        for tls in TLS_IDS:
            incoming_lanes = AGENTS[tls]["NS"] + AGENTS[tls]["EW"]
            total_v += sum(traci.lane.getLastStepVehicleNumber(l) for l in incoming_lanes)
    traci.close()

    # ✅ normalize per intersection
    avg_w = (total_w / MAX_STEPS) 
    avg_q = (total_q / MAX_STEPS) 
    avg_f = (total_f / MAX_STEPS)
    avg_v = (total_v / MAX_STEPS)

    return avg_w, avg_q, avg_f,avg_v

# =====================
# MAIN
# =====================

if __name__ == "__main__":

    waits, queues, flows,vehicle = [], [], [],[]

    for ep in range(EPISODES):
        w, q, f,v = run_episode()

        waits.append(w)
        queues.append(q)
        flows.append(f)
        vehicle.append(v)

        print(f"IQL | Ep {ep+1} | W:{w:.2f} Q:{q:.2f} F:{f:.2f}")

    # =====================
    # FINAL RESULTS
    # =====================

    avg_w = sum(waits) / EPISODES
    avg_q = sum(queues) / EPISODES
    avg_f = sum(flows) / EPISODES
    avg_v = sum(vehicle) / EPISODES

    print("\n===== IQL RESULTS =====")
    print(f"Avg W: {avg_w:.2f}")
    print(f"Avg Q: {avg_q:.2f}")
    print(f"Avg F: {avg_f:.2f}")
    print(f"Avg V: {avg_v:.2f}")

    # =====================
    # SAVE RESULTS
    # =====================

    results = {
        "waiting_times": waits,
        "queue_lengths": queues,
        "flows": flows,
        "avg_waiting": avg_w,
        "avg_queue": avg_q,
        "avg_flow": avg_f
    }

    file_path = os.path.join(RESULTS_DIR, f"IQL_eval_{EPISODES}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    print("💾 IQL evaluation results saved.")