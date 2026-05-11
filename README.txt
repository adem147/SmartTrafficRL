SmartTrafficRL
Overview

SmartTrafficRL is a reinforcement learning-based traffic signal control system developed using SUMO (Simulation of Urban Mobility). The project focuses on optimizing traffic flow at single and multi-intersection road networks by replacing fixed-time traffic lights with intelligent decision-making agents.

The system is designed to simulate realistic urban traffic scenarios and evaluate the performance of different reinforcement learning approaches such as Q-Learning and Deep Q-Networks (DQN). The goal is to reduce waiting time, queue length, and overall congestion in dynamic environments.

Project Objectives
Simulate realistic traffic environments using SUMO
Implement fixed-time, Q-Learning, and Deep Q-Learning controllers
Compare performance across single and multi-intersection setups
Reduce average waiting time and queue length
Improve throughput and traffic efficiency using adaptive control
System Architecture
SUMO simulation environment for traffic modeling
Python-based RL training framework
TraCI interface for communication with SUMO
Independent Q-Learning agents for multi-intersection control
Deep Q-Network for function approximation in larger state spaces
Traffic Control Approaches
Fixed-Time Control

Static traffic light cycles with predefined durations, independent of traffic conditions.

Q-Learning (Single and Multi-Intersection)

Tabular reinforcement learning where each intersection learns optimal actions based on state (queue length, waiting time).

Deep Q-Learning (DQN)

Neural network-based approach that approximates Q-values for more complex and continuous state representations.

Evaluation Metrics
Average waiting time
Average queue length
Throughput (vehicles passed)
Total delay
Reward convergence over episodes
Project Structure
SmartTrafficRL/
│
├── sumo/                  # SUMO network and configuration files
├── src/
│   ├── QL_training.py     # Q-Learning implementation
│   ├── DQN_training.py    # Deep Q-Learning implementation
│   ├── environment.py     # SUMO interaction logic
│
├── results/               # Logs, plots, and evaluation metrics
├── config/                # Simulation configuration files
└── README.md
Requirements
Python 3.8+
SUMO (Simulation of Urban Mobility)
TraCI
NumPy
TensorFlow or PyTorch (for DQN)
How to Run
Install SUMO and configure environment variables

Install Python dependencies:

pip install -r requirements.txt

Run training:

python src/QL_training.py

or

python src/DQN_training.py
Results

The reinforcement learning models are expected to outperform fixed-time control by:

Reducing congestion at intersections
Improving traffic flow adaptability
Decreasing overall vehicle waiting time
Future Work
Multi-agent coordination between intersections
Graph-based reinforcement learning approaches
Real-time deployment with IoT traffic sensors
Integration with emergency vehicle prioritization
Author

Developed by: Adem Riahi
Email: ademriahi94@gmail.com