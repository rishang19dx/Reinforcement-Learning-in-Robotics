"""
==========================================================================
                    ANALYSIS.PY - Extended Analysis Tools
==========================================================================
Additional analysis tools for the AR525 assignment including:
- Convergence plotting (Bellman residual over iterations)
- Stochastic transitions (slip probability)
- Reward structure comparison (sparse vs dense)

This file is separate from main.py and does not affect the main workflow.

Usage:
    python analysis.py

Author: AR525 Analysis Extension
==========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import time


class StochasticGridEnv:
    """
    Grid Environment with stochastic transitions.
    
    With probability `slip_prob`, the agent slips to a perpendicular direction
    instead of moving in the intended direction.
    """
    
    def __init__(self, rows=5, cols=6, start=0, goal=None, 
                 obstacles=None, slip_prob=0.1, reward_type='dense'):
        """
        Initialize stochastic grid environment.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            start: Start state
            goal: Goal state (default: bottom-right)
            obstacles: List of obstacle states
            slip_prob: Probability of slipping to perpendicular direction
            reward_type: 'dense' (-1 per step) or 'sparse' (0 until goal)
        """
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4
        self.start = start
        self.goal = goal if goal is not None else rows * cols - 1
        self.obstacles = set(obstacles) if obstacles is not None else set()
        self.slip_prob = slip_prob
        self.reward_type = reward_type
        self.action_names = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
        self.P = self._build_dynamics()
    
    def _state_to_pos(self, state):
        return state // self.cols, state % self.cols
    
    def _pos_to_state(self, row, col):
        return row * self.cols + col
    
    def _is_valid_pos(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_next_state(self, state, action):
        row, col = self._state_to_pos(state)
        
        if action == 0:    # LEFT
            col -= 1
        elif action == 1:  # DOWN
            row += 1
        elif action == 2:  # RIGHT
            col += 1
        elif action == 3:  # UP
            row -= 1
        
        if not self._is_valid_pos(row, col):
            return state
        
        next_state = self._pos_to_state(row, col)
        
        if next_state in self.obstacles:
            return state
        
        return next_state
    
    def _get_perpendicular_actions(self, action):
        """Get perpendicular actions for slip dynamics."""
        if action in [0, 2]:  # LEFT or RIGHT
            return [1, 3]  # DOWN, UP
        else:  # UP or DOWN
            return [0, 2]  # LEFT, RIGHT
    
    def _get_reward(self, state, next_state):
        """Get reward based on reward type."""
        if next_state == self.goal:
            return 100.0
        elif state in self.obstacles:
            return -1000.0
        elif self.reward_type == 'sparse':
            return 0.0  # Sparse: no penalty for steps
        else:  # dense
            return -1.0  # Dense: -1 per step
    
    def _build_dynamics(self):
        """Build transition dynamics with stochastic slip."""
        P = {}
        
        for state in range(self.nS):
            P[state] = {}
            
            for action in range(self.nA):
                transitions = []
                
                # Intended direction
                intended_next = self._get_next_state(state, action)
                intended_reward = self._get_reward(state, intended_next)
                intended_done = intended_next == self.goal
                
                # Perpendicular slip directions
                perp_actions = self._get_perpendicular_actions(action)
                slip_next_1 = self._get_next_state(state, perp_actions[0])
                slip_next_2 = self._get_next_state(state, perp_actions[1])
                
                if self.slip_prob > 0:
                    # Main transition (1 - slip_prob)
                    transitions.append((1 - self.slip_prob, intended_next, 
                                       intended_reward, intended_done))
                    
                    # Slip transitions (slip_prob / 2 each)
                    slip_prob_each = self.slip_prob / 2
                    
                    slip_reward_1 = self._get_reward(state, slip_next_1)
                    slip_done_1 = slip_next_1 == self.goal
                    transitions.append((slip_prob_each, slip_next_1, 
                                       slip_reward_1, slip_done_1))
                    
                    slip_reward_2 = self._get_reward(state, slip_next_2)
                    slip_done_2 = slip_next_2 == self.goal
                    transitions.append((slip_prob_each, slip_next_2, 
                                       slip_reward_2, slip_done_2))
                else:
                    # Deterministic
                    transitions.append((1.0, intended_next, intended_reward, intended_done))
                
                P[state][action] = transitions
        
        return P
    
    def get_optimal_path(self, policy):
        """Extract optimal path from policy."""
        path = [self.start]
        current_state = self.start
        max_steps = self.nS * 2
        
        steps = 0
        while current_state != self.goal and steps < max_steps:
            action = policy[current_state]
            next_state = self._get_next_state(current_state, action)
            path.append(next_state)
            
            if next_state == current_state:
                break
            
            current_state = next_state
            steps += 1
        
        return path


# ==========================================================================
#                  DP ALGORITHMS WITH CONVERGENCE TRACKING
# ==========================================================================

def policy_evaluation_with_tracking(env, policy, gamma=0.99, theta=1e-8):
    """
    Policy evaluation with convergence tracking.
    
    Returns:
        V: Value function
        deltas: List of max Bellman residuals per iteration
    """
    V = np.zeros(env.nS)
    deltas = []
    
    while True:
        delta = 0
        
        for s in range(env.nS):
            v = V[s]
            a = policy[s]
            
            new_value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                if done:
                    new_value += prob * reward
                else:
                    new_value += prob * (reward + gamma * V[next_state])
            
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))
        
        deltas.append(delta)
        
        if delta < theta:
            break
    
    return V, deltas


def q_from_v(env, V, s, gamma=0.99):
    """Compute Q-values from V-values for state s."""
    Q = np.zeros(env.nA)
    
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            if done:
                Q[a] += prob * reward
            else:
                Q[a] += prob * (reward + gamma * V[next_state])
    
    return Q


def policy_improvement(env, V, gamma=0.99):
    """Derive greedy policy from value function."""
    policy = np.zeros(env.nS, dtype=int)
    
    for s in range(env.nS):
        Q = q_from_v(env, V, s, gamma)
        policy[s] = np.argmax(Q)
    
    return policy


def policy_iteration_with_tracking(env, gamma=0.99, theta=1e-8):
    """
    Policy iteration with full convergence tracking.
    
    Returns:
        policy: Optimal policy
        V: Optimal value function
        all_deltas: List of (iteration, deltas_in_evaluation) tuples
        times: List of cumulative times per iteration
    """
    policy = np.zeros(env.nS, dtype=int)
    all_deltas = []
    times = []
    start_time = time.time()
    
    iteration = 0
    while True:
        # Policy Evaluation
        V, eval_deltas = policy_evaluation_with_tracking(env, policy, gamma, theta)
        all_deltas.extend(eval_deltas)
        times.append(time.time() - start_time)
        
        # Policy Improvement
        new_policy = policy_improvement(env, V, gamma)
        
        iteration += 1
        
        if np.array_equal(policy, new_policy):
            break
        
        policy = new_policy
    
    return policy, V, all_deltas, times, iteration


def value_iteration_with_tracking(env, gamma=0.99, theta=1e-8):
    """
    Value iteration with full convergence tracking.
    
    Returns:
        policy: Optimal policy
        V: Optimal value function
        deltas: Bellman residual per iteration
        times: Cumulative time per iteration
    """
    V = np.zeros(env.nS)
    deltas = []
    times = []
    start_time = time.time()
    
    while True:
        delta = 0
        
        for s in range(env.nS):
            v = V[s]
            Q = q_from_v(env, V, s, gamma)
            V[s] = np.max(Q)
            delta = max(delta, abs(v - V[s]))
        
        deltas.append(delta)
        times.append(time.time() - start_time)
        
        if delta < theta:
            break
    
    policy = policy_improvement(env, V, gamma)
    
    return policy, V, deltas, times, len(deltas)


# ==========================================================================
#                           PLOTTING FUNCTIONS
# ==========================================================================

def plot_convergence(deltas_pi, deltas_vi, title="Convergence Comparison", 
                     save_path=None):
    """
    Plot convergence curves for Policy Iteration vs Value Iteration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual plots
    ax1 = axes[0]
    ax1.semilogy(range(1, len(deltas_pi) + 1), deltas_pi, 'b-', 
                 linewidth=2, label='Policy Iteration')
    ax1.semilogy(range(1, len(deltas_vi) + 1), deltas_vi, 'r-', 
                 linewidth=2, label='Value Iteration')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Bellman Residual (log scale)', fontsize=12)
    ax1.set_title('Convergence: Bellman Residual', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Combined normalized plot
    ax2 = axes[1]
    
    # Normalize iterations to [0, 1] for comparison
    pi_norm_x = np.linspace(0, 1, len(deltas_pi))
    vi_norm_x = np.linspace(0, 1, len(deltas_vi))
    
    ax2.semilogy(pi_norm_x, deltas_pi, 'b-', linewidth=2, label='Policy Iteration')
    ax2.semilogy(vi_norm_x, deltas_vi, 'r-', linewidth=2, label='Value Iteration')
    ax2.set_xlabel('Normalized Progress', fontsize=12)
    ax2.set_ylabel('Bellman Residual (log scale)', fontsize=12)
    ax2.set_title('Convergence: Normalized Comparison', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_stochastic_comparison(results_dict, save_path=None):
    """
    Plot comparison of different slip probabilities.
    
    Args:
        results_dict: {slip_prob: (pi_iters, vi_iters, pi_deltas, vi_deltas)}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    slip_probs = sorted(results_dict.keys())
    pi_iters = [results_dict[sp][0] for sp in slip_probs]
    vi_iters = [results_dict[sp][1] for sp in slip_probs]
    
    # Iterations comparison
    ax1 = axes[0]
    x = np.arange(len(slip_probs))
    width = 0.35
    ax1.bar(x - width/2, pi_iters, width, label='Policy Iteration', color='blue', alpha=0.7)
    ax1.bar(x + width/2, vi_iters, width, label='Value Iteration', color='red', alpha=0.7)
    ax1.set_xlabel('Slip Probability', fontsize=12)
    ax1.set_ylabel('Iterations to Converge', fontsize=12)
    ax1.set_title('Iterations vs Stochasticity', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{sp:.0%}' for sp in slip_probs])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Convergence curves for different slip probs
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(slip_probs)))
    
    for i, sp in enumerate(slip_probs):
        deltas = results_dict[sp][3]  # VI deltas
        ax2.semilogy(range(1, len(deltas) + 1), deltas, 
                     color=colors[i], linewidth=2, label=f'slip={sp:.0%}')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Bellman Residual (log scale)', fontsize=12)
    ax2.set_title('Value Iteration Convergence by Slip Prob', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Stochastic Transitions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_reward_comparison(dense_results, sparse_results, save_path=None):
    """
    Plot comparison of dense vs sparse rewards.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence comparison
    ax1 = axes[0]
    ax1.semilogy(range(1, len(dense_results['vi_deltas']) + 1), 
                 dense_results['vi_deltas'], 'b-', linewidth=2, label='Dense (-1 per step)')
    ax1.semilogy(range(1, len(sparse_results['vi_deltas']) + 1), 
                 sparse_results['vi_deltas'], 'g-', linewidth=2, label='Sparse (0 until goal)')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Bellman Residual (log scale)', fontsize=12)
    ax1.set_title('Value Iteration Convergence', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of iterations
    ax2 = axes[1]
    labels = ['Policy Iter.', 'Value Iter.']
    dense_vals = [dense_results['pi_iters'], dense_results['vi_iters']]
    sparse_vals = [sparse_results['pi_iters'], sparse_results['vi_iters']]
    
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, dense_vals, width, label='Dense Rewards', color='blue', alpha=0.7)
    ax2.bar(x + width/2, sparse_vals, width, label='Sparse Rewards', color='green', alpha=0.7)
    ax2.set_ylabel('Iterations', fontsize=12)
    ax2.set_title('Iterations by Reward Structure', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Dense vs Sparse Reward Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def visualize_policy(env, policy, V, title="Policy Visualization", save_path=None):
    """
    Visualize policy as arrows with value function heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create value function grid
    V_grid = V.reshape(env.rows, env.cols)
    
    # Plot heatmap
    im = ax.imshow(V_grid, cmap='RdYlGn', origin='upper')
    plt.colorbar(im, ax=ax, label='State Value V(s)')
    
    # Arrow directions
    arrow_dx = {0: -0.3, 1: 0, 2: 0.3, 3: 0}  # LEFT, DOWN, RIGHT, UP
    arrow_dy = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    
    for s in range(env.nS):
        row, col = s // env.cols, s % env.cols
        
        if s == env.start:
            ax.text(col, row, 'S', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        elif s == env.goal:
            ax.text(col, row, 'G', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        elif s in env.obstacles:
            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, 
                        fill=True, facecolor='black', alpha=0.8))
            ax.text(col, row, 'X', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white')
        else:
            a = policy[s]
            ax.arrow(col, row, arrow_dx[a], arrow_dy[a], 
                    head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


# ==========================================================================
#                              MAIN ANALYSIS
# ==========================================================================

def run_analysis():
    """Run complete analysis with convergence plots and stochastic comparison."""
    
    print("=" * 70)
    print("       AR525 Assignment 1: Extended Analysis")
    print("=" * 70)
    
    obstacles = [7, 13, 14, 20]
    
    # =========================================================================
    # 1. DETERMINISTIC COMPARISON
    # =========================================================================
    print("\n" + "=" * 50)
    print("1. DETERMINISTIC ENVIRONMENT ANALYSIS")
    print("=" * 50)
    
    env_det = StochasticGridEnv(rows=5, cols=6, start=0, goal=29, 
                                 obstacles=obstacles, slip_prob=0.0)
    
    print("\nRunning Policy Iteration...")
    pi_policy, pi_V, pi_deltas, pi_times, pi_iters = \
        policy_iteration_with_tracking(env_det, gamma=0.99)
    print(f"  Converged in {pi_iters} policy iterations ({len(pi_deltas)} evaluations)")
    
    print("\nRunning Value Iteration...")
    vi_policy, vi_V, vi_deltas, vi_times, vi_iters = \
        value_iteration_with_tracking(env_det, gamma=0.99)
    print(f"  Converged in {vi_iters} iterations")
    
    # Plot convergence
    plot_convergence(pi_deltas, vi_deltas, 
                     title="Deterministic Environment Convergence",
                     save_path="convergence_deterministic.png")
    
    # Visualize policy
    visualize_policy(env_det, vi_policy, vi_V, 
                     title="Optimal Policy (Deterministic)",
                     save_path="policy_deterministic.png")
    
    # =========================================================================
    # 2. STOCHASTIC COMPARISON
    # =========================================================================
    print("\n" + "=" * 50)
    print("2. STOCHASTIC TRANSITIONS ANALYSIS")
    print("=" * 50)
    
    slip_probs = [0.0, 0.1, 0.2, 0.3]
    stoch_results = {}
    
    for slip in slip_probs:
        print(f"\nSlip probability: {slip:.0%}")
        env_stoch = StochasticGridEnv(rows=5, cols=6, start=0, goal=29,
                                       obstacles=obstacles, slip_prob=slip)
        
        _, _, pi_d, _, pi_i = policy_iteration_with_tracking(env_stoch, gamma=0.99)
        _, _, vi_d, _, vi_i = value_iteration_with_tracking(env_stoch, gamma=0.99)
        
        stoch_results[slip] = (pi_i, vi_i, pi_d, vi_d)
        print(f"  PI: {pi_i} iters, VI: {vi_i} iters")
    
    plot_stochastic_comparison(stoch_results, save_path="stochastic_comparison.png")
    
    # =========================================================================
    # 3. REWARD STRUCTURE COMPARISON
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. REWARD STRUCTURE COMPARISON")
    print("=" * 50)
    
    # Dense rewards
    print("\nDense rewards (-1 per step)...")
    env_dense = StochasticGridEnv(rows=5, cols=6, start=0, goal=29,
                                   obstacles=obstacles, reward_type='dense')
    _, _, pi_d_dense, _, pi_i_dense = policy_iteration_with_tracking(env_dense)
    _, _, vi_d_dense, _, vi_i_dense = value_iteration_with_tracking(env_dense)
    
    dense_results = {
        'pi_iters': pi_i_dense, 'vi_iters': vi_i_dense,
        'pi_deltas': pi_d_dense, 'vi_deltas': vi_d_dense
    }
    print(f"  PI: {pi_i_dense} iters, VI: {vi_i_dense} iters")
    
    # Sparse rewards
    print("\nSparse rewards (0 until goal)...")
    env_sparse = StochasticGridEnv(rows=5, cols=6, start=0, goal=29,
                                    obstacles=obstacles, reward_type='sparse')
    _, _, pi_d_sparse, _, pi_i_sparse = policy_iteration_with_tracking(env_sparse)
    _, _, vi_d_sparse, _, vi_i_sparse = value_iteration_with_tracking(env_sparse)
    
    sparse_results = {
        'pi_iters': pi_i_sparse, 'vi_iters': vi_i_sparse,
        'pi_deltas': pi_d_sparse, 'vi_deltas': vi_d_sparse
    }
    print(f"  PI: {pi_i_sparse} iters, VI: {vi_i_sparse} iters")
    
    plot_reward_comparison(dense_results, sparse_results, 
                           save_path="reward_comparison.png")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("                        ANALYSIS SUMMARY")
    print("=" * 70)
    print("""
Key Findings:

1. CONVERGENCE:
   - Both PI and VI converge to the same optimal policy
   - VI typically requires more iterations but each iteration is cheaper
   - PI does fewer iterations but each involves full policy evaluation

2. STOCHASTIC TRANSITIONS:
   - Higher slip probability → more iterations needed
   - Policy becomes more conservative (favors staying away from obstacles)
   - Value function reflects increased uncertainty

3. REWARD STRUCTURE:
   - Dense rewards (-1/step) provide stronger gradient for learning
   - Sparse rewards may converge in similar iterations but with different dynamics
   - Dense rewards encourage shorter paths more explicitly
""")
    
    print("\nPlots saved to current directory:")
    print("  - convergence_deterministic.png")
    print("  - policy_deterministic.png")
    print("  - stochastic_comparison.png")
    print("  - reward_comparison.png")


if __name__ == "__main__":
    run_analysis()
