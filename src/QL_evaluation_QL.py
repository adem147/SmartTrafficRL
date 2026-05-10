import traci
import random
import pickle
import os

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

EPISODES = 100   # ⬅️ smaller for evaluation
MAX_STEPS = 600

RESULTS_DIR = os.path.join(BASE_DIR, "results")
Q_TABLE_FILE = os.path.join(RESULTS_DIR, "q_table_2.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# LOAD Q-TABLE
# =====================

with open(Q_TABLE_FILE, "rb") as f:
    Q = pickle.load(f)

print("✅ Q-table loaded.")

# =====================
# Q-LEARNING PARAMETERS (EVALUATION MODE)
# =====================

epsilon = 0  # ❗ NO exploration

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

def collect_metrics():
    waiting = sum(traci.edge.getWaitingTime(e) for e in NS_EDGES + EW_EDGES)
    queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES + EW_EDGES)
    step_passed = sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES)
    return waiting, queue, step_passed

# =====================
# POLICY (NO EXPLORATION)
# =====================

ACTIONS = [PHASE_NS, PHASE_EW]

def choose_action(state):
    values = [Q.get((state, a), 0.0) for a in range(len(ACTIONS))]
    return int(max(range(len(ACTIONS)), key=lambda i: values[i]))

# =====================
# RUN EPISODE
# =====================

def run_episode():

    total_waiting = 0
    total_queue = 0
    total_passed = 0

    traci.start([SUMO_BINARY, "-c", SUMO_CFG, "--seed", str(random.randint(0, 1000000))])

    step = 0
    last_switch_step = -MIN_GREEN

    state = get_state()
    traci.trafficlight.setPhase(TLS_ID, PHASE_NS)
    
    total_v = 0

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        waiting, queue, step_passed = collect_metrics()

        total_waiting += waiting
        total_queue += queue
        total_passed += step_passed
        

        if step % T_DECISION == 0:

            action_idx = choose_action(state)
            chosen_phase = ACTIONS[action_idx]

            if step - last_switch_step >= MIN_GREEN:
                traci.trafficlight.setPhase(TLS_ID, chosen_phase)
                last_switch_step = step

            state = get_state()
            
       
        incoming_lanes = NS_EDGES + EW_EDGES
        total_v += sum(traci.edge.getLastStepVehicleNumber(l) for l in incoming_lanes)


    traci.close()

    avg_waiting = total_waiting / MAX_STEPS
    avg_queue = total_queue / MAX_STEPS
    avg_flow = total_passed / MAX_STEPS
    avg_v = total_v/MAX_STEPS

    return avg_waiting, avg_queue, avg_flow,avg_v

# =====================
# MAIN LOOP
# =====================

if __name__ == "__main__":

    waiting_times = []
    queue_lengths = []
    flows = []
    vehicle = []

    for ep in range(EPISODES):

        w, q, f,v = run_episode()

        waiting_times.append(w)
        queue_lengths.append(q)
        flows.append(f)
        vehicle.append(v)

        print(
            f"Episode {ep+1}/{EPISODES} | "
            f"Wait: {w:.2f} | "
            f"Queue: {q:.2f} | "
            f"Flow: {f:.2f}"
        )

    # =====================
    # FINAL AVERAGES
    # =====================

    avg_waiting = sum(waiting_times) / EPISODES
    avg_queue = sum(queue_lengths) / EPISODES
    avg_flow = sum(flows) / EPISODES
    avg_v = sum(vehicle) / EPISODES

    print("\n===== RL EVALUATION RESULTS =====")
    print(f"Avg Waiting Time: {avg_waiting:.2f}")
    print(f"Avg Queue Length: {avg_queue:.2f}")
    print(f"Avg Flow: {avg_flow:.2f}")
    print(f"Avg Vehicle: {avg_v:.2f}")
    
    # =====================
    # SAVE RESULTS
    # =====================

    results = {
        "waiting_times": waiting_times,
        "queue_lengths": queue_lengths,
        "flows": flows
    }

    file_path = os.path.join(RESULTS_DIR, f"QL_eval_results_{EPISODES}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    print("💾 RL evaluation results saved.")
