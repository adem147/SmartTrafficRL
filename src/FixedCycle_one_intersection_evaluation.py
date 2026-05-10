import traci
import random
import os
import pickle

# =====================
# CONFIG
# =====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_BINARY = "sumo"  # use "sumo-gui" to visualize
SUMO_CFG = os.path.join(BASE_DIR, "sumo", "simulation.sumocfg")

NS_EDGES = ["-E2", "-E3"]
EW_EDGES = ["-E1", "E0"]

# OUTGOING edges (vehicles leaving intersection)
OUT_EDGES = ["E1", "E2", "E3", "-E0"]

EPISODES = 100
MAX_STEPS = 600

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# METRICS COLLECTION
# =====================

def collect_metrics():
    waiting = sum(traci.edge.getWaitingTime(e) for e in NS_EDGES + EW_EDGES)
    queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES + EW_EDGES)
    flow = sum(traci.edge.getLastStepVehicleNumber(e) for e in OUT_EDGES)
    return waiting, queue, flow

# =====================
# RUN FIXED EPISODE
# =====================

def run_episode_fixed():

    total_waiting = 0
    total_queue = 0
    total_passed = 0

    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--seed", str(random.randint(0, 1000000))
    ])

    for step in range(MAX_STEPS):
        traci.simulationStep()

        waiting, queue, flow = collect_metrics()

        total_waiting += waiting
        total_queue += queue
        total_passed += flow

    traci.close()

    avg_waiting = total_waiting / MAX_STEPS
    avg_queue = total_queue / MAX_STEPS
    avg_flow = total_passed / MAX_STEPS

    return avg_waiting, avg_queue, avg_flow

# =====================
# MAIN LOOP
# =====================

if __name__ == "__main__":

    waiting_times = []
    queue_lengths = []
    flows = []

    for ep in range(EPISODES):

        w, q, f = run_episode_fixed()

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
    # FINAL AVERAGES
    # =====================

    avg_waiting = sum(waiting_times) / EPISODES
    avg_queue = sum(queue_lengths) / EPISODES
    avg_flow = sum(flows) / EPISODES

    print("\n===== FIXED TRAFFIC LIGHT RESULTS =====")
    print(f"Avg Waiting Time: {avg_waiting:.2f}")
    print(f"Avg Queue Length: {avg_queue:.2f}")
    print(f"Avg Flow (Throughput): {avg_flow:.2f}")

    # =====================
    # SAVE RESULTS
    # =====================

    results = {
        "waiting_times": waiting_times,
        "queue_lengths": queue_lengths,
        "flows": flows
    }

    file_path = os.path.join(RESULTS_DIR, f"fixed_eval_results_{EPISODES}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    print("💾fixed evaluation Results saved.")