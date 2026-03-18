# Assignment 2: Drone Hovering using Model-Free Control

## Overview

This assignment implements **Model-Free RL** algorithms to teach a drone to hover at a target position `[0, 0, 1]` in the [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) environment. The drone learns to stabilize using tabular Q-values over a discretized state space.

<p align="center">
  <img src="imgs/assignment_gif.gif" alt="Drone Hover Demo" width="500"/>
</p>

---

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
- [Algorithms Implemented](#algorithms-implemented)
- [Results & Analysis](#results--analysis)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Bonus Challenges](#bonus-challenges)
- [Code Structure](#code-structure)

---

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install gym-pybullet-drones (required)
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones && pip install -e . && cd ..

# Run training (MC + Q-Learning)
python user_code.py

# Evaluate submission
python evaluate_submission.py --student_file user_code.py --method all --seed 42 --min_reward 220 --eval_seeds 3

# Run bonus challenges
python bonus_challenges.py
```

---

## Environment Details

| Property | Value |
|----------|-------|
| **Environment** | `HoverAviary` (gym-pybullet-drones) |
| **Target Position** | `[0, 0, 1]` (hover at z=1.0) |
| **Observation** | Kinematic state (position, velocity, orientation) |
| **Action** | `ONE_D_RPM` — thrust adjustment `{-1, 0, +1}` |
| **Episode Length** | 240 steps (8 seconds at 30 Hz) |
| **Reward** | Proximity-based: higher reward for being closer to target |

<p align="center">
  <img src="imgs/Screenshot from 2026-02-11 23-39-32.png" alt="PyBullet Simulation Environment" width="500"/>
</p>

### State Discretization

The continuous 3D position `(x, y, z)` is discretized into `NUM_BINS=10` bins per dimension, producing a `10×10×10×3` Q-table (3000 state-action pairs).

| Dimension | Range | Bins |
|-----------|-------|------|
| x | [-1, 1] | 10 |
| y | [-1, 1] | 10 |
| z | [0, 2]  | 10 |

<p align="center">
  <img src="imgs/state_discretization.svg" alt="State Discretization Grid" width="450"/>
</p>

<p align="center">
  <img src="imgs/state_discretization.gif" alt="State Discretization Animation" width="450"/>
</p>

---

## Algorithms Implemented

### 1. Monte Carlo Control (First-Visit)

Generates complete episodes using ε-greedy exploration, computes discounted returns backward from the terminal step, and updates Q-values for first-visit state-action pairs.

**Update Rule:** `Q(s,a) ← Q(s,a) + α [G - Q(s,a)]`

```
For each episode:
  1. Generate full trajectory using ε-greedy policy
  2. Walk backward: G = γ·G + r_t
  3. For first-visit (s,a) pairs: Q(s,a) += α·(G - Q(s,a))
```

**Key Characteristics:**
- Learns from *complete* episodes (no bootstrapping)
- High variance but unbiased estimates
- Requires episode termination before updates

### 2. Q-Learning (Off-Policy TD Control)

Updates Q-values at each timestep using the Bellman optimality equation. Off-policy: uses the max Q-value of the next state regardless of the action actually taken.

**Update Rule:** `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]`

```
At each step:
  1. Take action a (ε-greedy), observe r, s'
  2. TD target = r + γ · max Q(s', ·)
  3. Q(s,a) += α · (target - Q(s,a))
```

**Key Characteristics:**
- Online updates (no need to wait for episode end)
- Bootstraps from estimated values (lower variance, some bias)
- Converges to optimal policy under sufficient exploration

---

## Results & Analysis

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| `NUM_BINS` | 10 |
| `EPSILON` (ε) | 0.1 |
| `GAMMA` (γ) | 0.99 |
| `ALPHA` (α) | 0.1 |
| `NUM_EPISODES` | 500 |
| `MAX_STEPS` | 240 |

### MC vs. Q-Learning Comparison

| Metric | Monte Carlo | Q-Learning |
|--------|-------------|------------|
| **Update Timing** | End of episode | Every step |
| **Bias/Variance** | Unbiased, high variance | Biased (bootstrap), low variance |
| **Convergence Speed** | Slower (needs full episodes) | Faster (online updates) |
| **Exploration** | Full episode before update | Immediate feedback |
| **Policy Type** | On-policy (ε-greedy) | Off-policy (learns greedy via ε-greedy) |

### Key Findings

1. **Q-Learning converges faster** because it updates after every step, while MC must wait for the full 240-step episode before learning.

2. **MC has higher variance** in early episodes — the drone's initial random exploration produces wildly different returns, making Q-value estimates noisy.

3. **Both methods eventually learn** to hover near z=1.0 with the given hyperparameters. The drone stabilizes its thrust adjustment to maintain altitude.

4. **State discretization** is critical — too few bins (< 8) leads to coarse policies; too many bins (> 15) increases the state space exponentially and requires more episodes to fill the Q-table.

### Simulation Results

Recorded drone behavior after training:

<p align="center">
  <img src="imgs/new.gif" alt="Trained Drone Hovering" width="400"/>
</p>

---

## Hyperparameter Tuning

We used **Optuna** (Bayesian optimization with pruning) to find optimal hyperparameters for Q-Learning. The search space and findings are below.

### Search Space

| Parameter | Range | Best Found |
|-----------|-------|------------|
| `NUM_BINS` | 8–15 | ~10 |
| `EPSILON` | 0.05–0.3 | ~0.1 |
| `GAMMA` | 0.95–0.999 | ~0.99 |
| `ALPHA` | 0.05–0.2 | ~0.1 |
| `NUM_EPISODES` | 500–1000 | ~500 |

### Observations

- **`GAMMA` ≈ 0.99** works best — high discount factor is needed because the hover task requires long-term planning over 240 steps.
- **`EPSILON` ≈ 0.1** balances exploration vs. exploitation. Lower values converge faster but risk getting stuck; higher values explore too much and destabilize.
- **`ALPHA` ≈ 0.1** provides stable learning. Higher learning rates cause Q-value oscillation, while lower rates slow convergence.
- **`NUM_BINS` ≈ 10** is the sweet spot. The 10³ = 1000 states are reachable within 500 episodes while providing sufficient granularity.

### Running HPO

```bash
python optimize_hpo.py  # Runs 30 Optuna trials with median pruning
```

---

## Bonus Challenges

All three bonus challenges were implemented in `bonus_challenges.py`:

### Challenge 1: SARSA (5 pts) ⭐

**On-policy TD control** — uses the *actual next action* from the ε-greedy policy in the update, making it more conservative than Q-Learning.

**Update:** `Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]`

> Unlike Q-Learning's `max Q(s',·)`, SARSA uses the Q-value of the action actually chosen. This means SARSA accounts for exploration in its value estimates, learning a safer policy.

### Challenge 2: Double Q-Learning (7 pts) ⭐⭐

**Reduces maximization bias** in standard Q-Learning by maintaining two independent Q-tables (Q1, Q2) and alternating updates.

```
With 50% probability:
  Update Q1: use argmax(Q2) for evaluation
Otherwise:
  Update Q2: use argmax(Q1) for evaluation
Action selection: ε-greedy over (Q1 + Q2) / 2
```

> Standard Q-Learning overestimates Q-values because `max` is a biased estimator. By decoupling action selection and evaluation across two tables, Double Q-Learning produces more accurate value estimates.

### Challenge 3: Experience Replay (8 pts) ⭐⭐⭐

**Stores transitions** `(s, a, r, s', done)` in a replay buffer (capacity=10,000) and samples random mini-batches for learning.

```
1. Collect experience → push to buffer
2. Sample random batch (size=32)
3. Perform Q-Learning update on each sample
```

> **Benefits:** Breaks temporal correlation between consecutive experiences, enables reuse of past data, and significantly improves learning stability and sample efficiency.

### Running Bonus Challenges

```bash
python bonus_challenges.py  # Trains and evaluates all 3 bonus algorithms
```

---

## Code Structure

```
a2/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── user_code.py               # ✅ Monte Carlo + Q-Learning implementation
├── bonus_challenges.py        # ✅ SARSA, Double Q-Learning, Experience Replay
├── optimize_hpo.py            # ✅ Optuna hyperparameter optimization
├── evaluate_submission.py     # Automated grading script
├── imgs/
│   ├── assignment_gif.gif     # Target hover behavior demo
│   ├── new.gif                # Trained drone result
│   ├── state_discretization.svg/gif  # State space visualization
│   └── Screenshot*.png        # Environment screenshot
└── results/
    ├── recording_*/           # Frame-by-frame recordings
    └── video-*.mp4            # Simulation videos
```

---

## Grading

| Component | Weight | Status |
|-----------|--------|--------|
| Monte Carlo Control | 30% | ✅ Implemented |
| Q-Learning (TD Control) | 30% | ✅ Implemented |
| Experiments & Analysis | 25% | ✅ HPO + comparison |
| Code Quality | 15% | ✅ Documented |
| **Bonus: SARSA** | +5 pts | ✅ Complete |
| **Bonus: Double Q-Learning** | +7 pts | ✅ Complete |
| **Bonus: Experience Replay** | +8 pts | ✅ Complete |

### Evaluate

```bash
python evaluate_submission.py --student_file user_code.py --method all --seed 42 --min_reward 220 --eval_seeds 3
```

---

## References

- [Sutton & Barto — Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
- [Monte Carlo Methods — Ch. 5](http://incompleteideas.net/book/ebook/node49.html)
- [Q-Learning — Ch. 6](http://incompleteideas.net/book/ebook/node47.html)
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

## Acknowledgements

Thanks to Sadbhav Singh [(@sadbhavsingh16)](https://github.com/sadbhavsingh16) for his help in preparing this assignment.
