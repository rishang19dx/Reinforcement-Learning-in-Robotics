# AR525: Reinforcement Learning in Robotics

This repository contains assignments for the course **AR525: Reinforcement Learning in Robotics**.

## Assignments

| Assignment | Topic | Status |
|------------|-------|--------|
| [A1](./a1/) | Dynamic Programming Algorithms | ✅ Complete |
| [A2](./a2/) | Model-Free Control (Monte Carlo & Q-Learning) | ✅ Complete |

---

## A1: Grid Navigation using Dynamic Programming

Implements **Policy Iteration** and **Value Iteration** for robotic path planning. A UR5 manipulator navigates a grid world with obstacles using optimal policies computed via Dynamic Programming.

**Key Features:**
- Policy Evaluation, Q-value computation, Policy Improvement
- Full Policy Iteration and Value Iteration algorithms
- Obstacle avoidance and flexible grid sizes
- PyBullet simulation with robot visualization
- Extended analysis with stochastic transitions and convergence plots

```bash
cd a1
python main.py      # Run simulation
python analysis.py  # Generate analysis plots
```

---

## A2: Drone Hovering using Model-Free Control

Implements **Monte Carlo Control** and **Q-Learning** to teach a drone to hover at a target position. The drone learns to stabilize at `[0, 0, 1]` using tabular RL with state discretization in a PyBullet drone simulator.

**Key Features:**
- First-visit Monte Carlo Control with epsilon-greedy exploration
- Off-policy Q-Learning (TD Control)
- State discretization of continuous 3D position into discrete bins
- Hyperparameter tuning and learning curve analysis
- Bonus challenges: SARSA, Double Q-Learning, Experience Replay

```bash
cd a2
pip install -r requirements.txt
python user_code.py                # Run student implementation
python evaluate_submission.py \
  --student_file user_code.py \
  --method all --seed 42           # Evaluate submission
python bonus_challenges.py         # Run bonus challenges
```

---

## Course Information

- **Course:** AR525 - Reinforcement Learning in Robotics
- **Semester:** Jan 2026 (6th Semester)
