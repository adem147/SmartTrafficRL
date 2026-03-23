import traci
import random
import pickle
import os


# CONFIG

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_BINARY = "sumo"           # use "sumo-gui" for visualization
SUMO_CFG = "simulation.sumocfg"
SUMO_DIR = os.path.join(BASE_DIR, "sumo")
SUMO_CFG = os.path.join(SUMO_DIR, SUMO_CFG)

NS_EDGES = ["-E2", "-E3"]
EW_EDGES = ["-E1", "E0"]

PHASE_NS = 0
PHASE_EW = 2
TLS_ID = "J1"

T_DECISION = 10
MIN_GREEN = 5

EPISODES = 10
MAX_STEPS = 600
SAVE_EVERY = 10

Q_TABLE_FILE = "q_table_2.pkl"
RESULTS_DIR = os.path.join(BASE_DIR,"results")
Q_TABLE_FILE = os.path.join(RESULTS_DIR, Q_TABLE_FILE)




# Q-LEARNING PARAMETERS


alpha = 0.1
gamma = 0.95
epsilon = 1
epsilon_min = 0.05
epsilon_decay = 0.999


# STATE DISCRETIZATION

BINS = [0, 3,6,9,12]  # 0, 1-3, 4-6,7-9,10-12,13+

def to_bin(x):
    for i, b in enumerate(BINS):
        if x <= b:
            return i
    return len(BINS)

def get_state():
    ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in NS_EDGES)
    ew = sum(traci.edge.getLastStepHaltingNumber(e) for e in EW_EDGES)
    return (to_bin(ns), to_bin(ew))

def get_reward():
    waiting =  (sum(traci.edge.getWaitingTime(e) for e in NS_EDGES) * 1.0 +
            sum(traci.edge.getWaitingTime(e) for e in EW_EDGES) * 1.0)
    return - (waiting)


# LOAD OR CREATE Q-TABLE





if os.path.exists(Q_TABLE_FILE):
    print("File exists")

if os.path.exists(Q_TABLE_FILE):
    with open(Q_TABLE_FILE, "rb") as f:
        Q = pickle.load(f)
    print("✅ Q-table loaded.")
else:
    Q = {}
    print("🆕 New Q-table created.")

ACTIONS = [PHASE_NS, PHASE_EW]


# Q FUNCTIONS


def choose_action(state):
    if random.random() < epsilon:
        return random.choice(range(len(ACTIONS)))
    values = [Q.get((state, a), 0.0) for a in range(len(ACTIONS))]
    return int(max(range(len(ACTIONS)), key=lambda i: values[i]))

def update_q(state, action, reward, next_state):
    key = (state, action)
    qsa = Q.get(key, 0.0)
    next_vals = [Q.get((next_state, a), 0.0) for a in range(len(ACTIONS))]
    Q[key] = qsa + alpha * (reward + gamma * max(next_vals) - qsa)


# TRAINING EPISODE


def run_episode():
    global epsilon

    traci.start([SUMO_BINARY, "-c", SUMO_CFG])

    step = 0
    last_switch_step = -MIN_GREEN
    state = get_state()
    total_reward = 0

    traci.trafficlight.setPhase(TLS_ID, PHASE_NS)

    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1

        if step % T_DECISION == 0:
            state = get_state()
            action_idx = choose_action(state)
            chosen_phase = ACTIONS[action_idx]

            if (step - last_switch_step >= MIN_GREEN and
                traci.trafficlight.getPhase(TLS_ID) != chosen_phase):
                traci.trafficlight.setPhase(TLS_ID, chosen_phase)
                last_switch_step = step

            next_state = get_state()
            reward = get_reward()
            total_reward += reward

            update_q(state, action_idx, reward, next_state)
            state = next_state

    traci.close()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return total_reward


# TRAIN LOOP


if __name__ == "__main__":
    rewards_per_episode = []

    for ep in range(EPISODES):
        episode_reward = run_episode()
        
        rewards_per_episode.append(episode_reward)

        print(f"Episode {ep+1}/{EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")
        
        # Save periodically
        if (ep + 1) % SAVE_EVERY == 0:
            with open(Q_TABLE_FILE, "wb") as f:
                pickle.dump(Q, f)
            print(f"💾 Q-table saved at episode {ep+1}")

         

    # Save rewards AFTER training
    
    TrainingDataFileName = "trainingData_"+str(EPISODES)+".pkl"
    TrainingDataFileName = os.path.join(RESULTS_DIR,TrainingDataFileName)
    
    with open(TrainingDataFileName, "wb") as f:
        pickle.dump(rewards_per_episode, f)

    print("📊 Rewards saved.")

       
    # Final save
    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(Q, f)

    print("🎉 Training complete. Final Q-table saved.")
