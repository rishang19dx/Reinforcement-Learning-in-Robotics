"""
==========================================================================
                        UTILS.PY - STUDENT IMPLEMENTATION
==========================================================================
Students must implement the Dynamic Programming algorithms below.

Author: Assignment 1 - AR525
==========================================================================
"""

import numpy as np

class GridEnv:
    
    def __init__(self, rows=5, cols=6, start=0, goal=29, obstacles=None):
        """
        Initialize GridEnv with flexible grid size and optional obstacles.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            start: Starting state (default: 0, top-left)
            goal: Goal state (default: rows*cols-1, bottom-right)
            obstacles: List/set of obstacle state indices (optional)
        """
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4
        self.start = start
        self.goal = goal if goal is not None else rows * cols - 1
        self.obstacles = set(obstacles) if obstacles is not None else set()
        self.action_names = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
        self.P = self._build_dynamics()
    
    def _state_to_pos(self, state):

        return state // self.cols, state % self.cols
    
    def _pos_to_state(self, row, col):

        return row * self.cols + col
    
    def _is_valid_pos(self, row, col):

        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_next_state(self, state, action):
        """
        Get next state given current state and action.
        Handles grid boundaries and obstacle collisions.
        """
        row, col = self._state_to_pos(state)
        
        if action == 0:    # LEFT
            col -= 1
        elif action == 1:  # DOWN
            row += 1
        elif action == 2:  # RIGHT
            col += 1
        elif action == 3:  # UP
            row -= 1
        
        # Check grid boundaries
        if not self._is_valid_pos(row, col):
            return state
        
        next_state = self._pos_to_state(row, col)
        
        # Check obstacle collision - stay in place if hitting obstacle
        if next_state in self.obstacles:
            return state
        
        return next_state
    
    def _build_dynamics(self):

        P = {}
        
        for state in range(self.nS):
            P[state] = {}
            
            for action in range(self.nA):
                next_state = self._get_next_state(state, action)
                
                # ============================================================
                # Reward Structure:
                # - Goal state: +100 reward
                # - Obstacle states: Large negative reward (should not be reachable)
                # - All other states: -1 reward (encourages shortest path)
                # ============================================================
              
                if next_state == self.goal:
                    reward = 100.0
                    done = True
                elif state in self.obstacles:
                    # This state is an obstacle - should not be visited
                    reward = -1000.0
                    done = False
                else:
                    reward = -1.0
                    done = False
                
                P[state][action] = [(1.0, next_state, reward, done)]
        
        return P
    
    def get_optimal_path(self, policy):
        """
        Extract the optimal path from start to goal using the given policy.
        
        Args:
            policy: Array of shape (nS,) containing optimal action for each state
            
        Returns:
            path: List of states from start to goal
        """
        path = [self.start]
        current_state = self.start
        max_steps = self.nS * 2  # Prevent infinite loops
        
        steps = 0
        while current_state != self.goal and steps < max_steps:
            action = policy[current_state]
            next_state = self._get_next_state(current_state, action)
            path.append(next_state)
            
            # Check if stuck (no progress)
            if next_state == current_state:
                break
            
            current_state = next_state
            steps += 1
        
        return path


# ==========================================================================
#                  DYNAMIC PROGRAMMING ALGORITHMS
# ==========================================================================

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Evaluate a policy using iterative policy evaluation.
    
    Implements the Bellman expectation equation:
    V(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a)[r + γV(s')]
    
    Args:
        env: GridEnv environment with dynamics P[s][a] = [(prob, next_state, reward, done), ...]
        policy: Array of shape (nS,) where policy[s] is the action to take in state s
        gamma: Discount factor (default: 0.99)
        theta: Convergence threshold (default: 1e-8)
        
    Returns:
        V: Array of shape (nS,) representing state-value function
    """
    # Initialize value function to zeros
    V = np.zeros(env.nS)
    
    iteration = 0
    while True:
        delta = 0
        
        # Sweep through all states
        for s in range(env.nS):
            v = V[s]  # Store old value
            
            # Get action from policy (deterministic)
            a = policy[s]
            
            # Compute new value using Bellman expectation equation
            new_value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                if done:
                    new_value += prob * reward
                else:
                    new_value += prob * (reward + gamma * V[next_state])
            
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))
        
        iteration += 1
        
        # Check for convergence
        if delta < theta:
            break
    
    return V


def q_from_v(env, V, s, gamma=0.99):
    """
    Compute action-value function Q(s,a) from state-value function V(s) for a given state.
    
    Implements: Q(s,a) = Σ_{s',r} P(s',r|s,a)[r + γV(s')]
    
    Args:
        env: GridEnv environment
        V: State-value function array of shape (nS,)
        s: State to compute Q-values for
        gamma: Discount factor (default: 0.99)
        
    Returns:
        Q: Array of shape (nA,) containing Q-values for each action in state s
    """
    Q = np.zeros(env.nA)
    
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            if done:
                Q[a] += prob * reward
            else:
                Q[a] += prob * (reward + gamma * V[next_state])
    
    return Q


def policy_improvement(env, V, gamma=0.99):
    """
    Improve a policy by making it greedy with respect to the value function.
    
    Implements: π(s) = argmax_a Q(s,a)
    
    Args:
        env: GridEnv environment
        V: State-value function array of shape (nS,)
        gamma: Discount factor (default: 0.99)
        
    Returns:
        policy: Improved policy array of shape (nS,)
    """
    policy = np.zeros(env.nS, dtype=int)
    
    for s in range(env.nS):
        # Compute Q-values for all actions in state s
        Q = q_from_v(env, V, s, gamma)
        
        # Select the action with maximum Q-value (greedy)
        policy[s] = np.argmax(Q)
    
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Find optimal policy using Policy Iteration.
    
    Alternates between:
    1. Policy Evaluation: Compute V^π for current policy π
    2. Policy Improvement: Compute new greedy policy π' from V^π
    Repeat until policy converges.
    
    Args:
        env: GridEnv environment
        gamma: Discount factor (default: 0.99)
        theta: Convergence threshold for policy evaluation (default: 1e-8)
        
    Returns:
        policy: Optimal policy array of shape (nS,)
        V: Optimal state-value function array of shape (nS,)
        iterations: Number of policy improvement iterations
    """
    # Initialize random policy
    policy = np.zeros(env.nS, dtype=int)
    
    iterations = 0
    while True:
        # Policy Evaluation: compute value function for current policy
        V = policy_evaluation(env, policy, gamma, theta)
        
        # Policy Improvement: compute greedy policy from value function
        new_policy = policy_improvement(env, V, gamma)
        
        iterations += 1
        
        # Check if policy has converged (no change)
        if np.array_equal(policy, new_policy):
            break
        
        policy = new_policy
    
    return policy, V, iterations


def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Find optimal policy using Value Iteration.
    
    Combines policy evaluation and improvement into a single update:
    V(s) = max_a Σ_{s',r} P(s',r|s,a)[r + γV(s')]
    
    Args:
        env: GridEnv environment
        gamma: Discount factor (default: 0.99)
        theta: Convergence threshold (default: 1e-8)
        
    Returns:
        policy: Optimal policy array of shape (nS,)
        V: Optimal state-value function array of shape (nS,)
        iterations: Number of value iterations
    """
    # Initialize value function to zeros
    V = np.zeros(env.nS)
    
    iterations = 0
    while True:
        delta = 0
        
        # Sweep through all states
        for s in range(env.nS):
            v = V[s]  # Store old value
            
            # Compute Q-values for all actions
            Q = q_from_v(env, V, s, gamma)
            
            # Take maximum Q-value (Bellman optimality)
            V[s] = np.max(Q)
            
            delta = max(delta, abs(v - V[s]))
        
        iterations += 1
        
        # Check for convergence
        if delta < theta:
            break
    
    # Extract optimal policy from converged value function
    policy = policy_improvement(env, V, gamma)
    
    return policy, V, iterations
