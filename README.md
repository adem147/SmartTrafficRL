# SmartTrafficRL

## Overview

SmartTrafficRL is a reinforcement learning-based traffic signal control system developed using SUMO (Simulation of Urban Mobility).

The project focuses on optimizing traffic flow in single and multi-intersection road networks by replacing fixed-time traffic lights with adaptive decision-making agents.

The system simulates urban traffic scenarios and evaluates reinforcement learning approaches such as Q-Learning and Deep Q-Networks (DQN) to reduce congestion and improve efficiency.

---

## Project Objectives

- Simulate traffic environments using SUMO  
- Implement fixed-time, Q-Learning, and Deep Q-Learning controllers  
- Compare performance across single and multi-intersection setups  
- Reduce average waiting time and queue length  
- Improve throughput using adaptive control  

---

## System Architecture

- SUMO simulation environment for traffic modeling  
- Python-based reinforcement learning framework  
- TraCI interface for SUMO communication  
- Independent Q-Learning agents for multi-intersection control  
- Deep Q-Network for function approximation in large state spaces  

---

## Traffic Control Approaches

### Fixed-Time Control
Static traffic light cycles with predefined durations, independent of traffic conditions.

### Q-Learning (Single and Multi-Intersection)
Tabular reinforcement learning where each intersection learns optimal actions based on state variables such as queue length and waiting time.

### Deep Q-Learning (DQN)
Neural network-based method that approximates Q-values for more complex state representations.

---

## Evaluation Metrics

- Average waiting time  
- Average queue length  
- Throughput (vehicles passed)  
- Total delay  
- Reward convergence over episodes  

---

## Requirements

- Python 3.8+
- SUMO (Simulation of Urban Mobility)
- TraCI
- NumPy
- TensorFlow or PyTorch (for DQN)

---

## How to Run

### Install SUMO and configure environment variables

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run training

Q-Learning:
```bash
python src/QL_training.py
```

Deep Q-Learning:
```bash
python src/DQN_training.py
```

---

## Results

The reinforcement learning models are expected to outperform fixed-time control by:

- Reducing congestion at intersections  
- Improving traffic flow adaptability  
- Decreasing overall vehicle waiting time  

---

## Future Work

- Multi-agent coordination between intersections  
- Graph-based reinforcement learning approaches  
- Real-time deployment with IoT traffic sensors  
- Integration with emergency vehicle prioritization  

---

## Author

Adem Riahi  
ademriahi94@gmail.com
bechir chemem 
feten ochi 
