import os
import random
import pickle
import traci

# =====================
# CONFIG
# =====================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SUMO_BINARY = "sumo"
SUMO_CFG    = os.path.join(BASE_DIR, "sumo", "simulation_2.sumocfg")

TLS_IDS = ["J1", "J5"]

AGENTS = {
    "J1": {"NS": ["E2_0", "E3_0"], "EW": ["E1_0", "E0_0"]},
    "J5": {"NS": ["E5_0", "E7_0"], "EW": ["E6_0", "-E4_0"]}
}

OUT_EDGES = {
    "J1": ["-E0", "-E1", "-E2", "-E3"],
    "J5": ["E4", "-E5", "-E6", "-E7"]
}

# Cycle defined in intersection_2.net.xml (type="static"):
#   Phase 0 — NS green  : 30 steps
#   Phase 1 — NS yellow :  3 steps
#   Phase 2 — EW green  : 30 steps
#   Phase 3 — EW yellow :  3 steps
# SUMO runs this automatically — no traci phase control needed.

MAX_STEPS = 600
EPISODES  = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

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
# EPISODE
# =====================

def run_episode():

    traci.start([SUMO_BINARY, "-c", SUMO_CFG,
                 "--seed", str(random.randint(0, 1000000))])

    # No phase overrides — SUMO runs the static tlLogic from the net file directly.

    step = 0
    total_w, total_q, total_f = 0, 0, 0

    while step < MAX_STEPS:

        traci.simulationStep()
        step += 1

        w, q, f = collect_metrics()
        total_w += w
        total_q += q
        total_f += f

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
            f"Fixed | Ep {ep+1}/{EPISODES} | "
            f"W: {w:.2f} | Q: {q:.2f} | F: {f:.2f}"
        )

    # =====================
    # FINAL RESULTS
    # =====================

    avg_waiting = sum(waits) / EPISODES
    avg_queue   = sum(queues) / EPISODES
    avg_flow    = sum(flows) / EPISODES

    print("\n===== FIXED CYCLE RESULTS (2 intersections) =====")
    print(f"Avg Waiting : {avg_waiting:.2f}")
    print(f"Avg Queue   : {avg_queue:.2f}")
    print(f"Avg Flow    : {avg_flow:.2f}")

    # =====================
    # SAVE RESULTS
    # =====================

    results = {
        "waiting_times" : waits,
        "queue_lengths" : queues,
        "flows"         : flows,
        "avg_waiting"   : avg_waiting,
        "avg_queue"     : avg_queue,
        "avg_flow"      : avg_flow
    }

    file_path = os.path.join(RESULTS_DIR, f"fixed_cycle_eval_twoIntersections{EPISODES}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    print("💾 Fixed cycle evaluation results saved.")