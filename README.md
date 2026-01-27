# AR525: Reinforcement Learning in Robotics

This repository contains assignments for the course **AR525: Reinforcement Learning in Robotics**.

## Assignments

| Assignment | Topic | Status |
|------------|-------|--------|
| [A1](./a1/) | Dynamic Programming Algorithms | ✅ Complete |

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

## Course Information

- **Course:** AR525 - Reinforcement Learning in Robotics
- **Semester:** Jan 2026 (6th Semester)
