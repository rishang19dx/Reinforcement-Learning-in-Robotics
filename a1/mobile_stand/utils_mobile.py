"""
==========================================================================
                  UTILS.PY - MOBILE MANIPULATOR VERSION
==========================================================================
Enhanced version where the end-effector must reach goals, with mobile
base moving to assist when targets are out of arm's reach.

Author: Assignment 1 - AR525 (Mobile Manipulator Extension)
==========================================================================
"""

import numpy as np

class MobileManipulatorEnv:
    
    def __init__(self, rows=10, cols=10, start=0, goal=None, obstacles=None, 
                 arm_reach=1.0, grid_size=0.20):
        """
        Initialize environment for mobile manipulator.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            start: Starting state (mobile base position)
            goal: Goal state (where end-effector must reach)
            obstacles: List/set of obstacle state indices
            arm_reach: Maximum reach of arm in grid cells (default: 2.0 cells)
            grid_size: Size of each grid cell in meters
        """
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4  # LEFT, DOWN, RIGHT, UP
        self.start = start
        self.goal = goal if goal is not None else rows * cols - 1
        self.obstacles = set(obstacles) if obstacles is not None else set()
        self.action_names = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
        
        # Mobile manipulator specific
        self.arm_reach = arm_reach  # in grid cells
        self.grid_size = grid_size
        
        self.P = self._build_dynamics()
    
    def _state_to_pos(self, state):
        """Convert state index to (row, col) position."""
        return state // self.cols, state % self.cols
    
    def _pos_to_state(self, row, col):
        """Convert (row, col) position to state index."""
        return row * self.cols + col
    
    def _is_valid_pos(self, row, col):
        """Check if position is within grid boundaries."""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _distance(self, state1, state2):
        """Calculate Euclidean distance between two states in grid cells."""
        r1, c1 = self._state_to_pos(state1)
        r2, c2 = self._state_to_pos(state2)
        return np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
    
    def can_reach_goal_from_state(self, state):
        """Check if end-effector can reach goal from given base state."""
        return self._distance(state, self.goal) <= self.arm_reach
    
    def _get_next_state(self, state, action):
        """Get next state given current state and action."""
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
        
        # Check obstacle collision
        if next_state in self.obstacles:
            return state
        
        return next_state
    
    def _build_dynamics(self):
        """Build transition dynamics for mobile manipulator."""
        P = {}
        
        for state in range(self.nS):
            P[state] = {}
            
            for action in range(self.nA):
                next_state = self._get_next_state(state, action)
                
                # ============================================================
                # Reward Structure for Mobile Manipulator:
                # - Goal reachable: +100 (end-effector can reach goal)
                # - Getting closer to goal: small positive reward
                # - Moving away from goal: negative reward
                # - Movement cost: -1 per step
                # - Wall/obstacle collision: -2
                # ============================================================
                
                # Check if end-effector can reach goal from next state
                can_reach = self.can_reach_goal_from_state(next_state)
                
                if can_reach:
                    # Success! End-effector can reach goal from here
                    reward = 100.0
                    done = True
                elif state in self.obstacles:
                    # Should not be in obstacle
                    reward = -1000.0
                    done = False
                elif next_state == state:
                    # Hit wall or obstacle - stayed in place
                    reward = -2.0
                    done = False
                else:
                    # Normal movement - reward based on getting closer to goal
                    old_dist = self._distance(state, self.goal)
                    new_dist = self._distance(next_state, self.goal)
                    
                    if new_dist < old_dist:
                        # Moving closer to goal
                        reward = -1.0 + 0.5  # Base cost + progress bonus
                    else:
                        # Moving away or same distance
                        reward = -1.0
                    
                    done = False
                
                P[state][action] = [(1.0, next_state, reward, done)]
        
        return P
    
    def get_optimal_path(self, policy):
        """Extract optimal path from start to a state where goal is reachable."""
        path = [self.start]
        current_state = self.start
        max_steps = self.nS * 2
        
        steps = 0
        while not self.can_reach_goal_from_state(current_state) and steps < max_steps:
            action = policy[current_state]
            next_state = self._get_next_state(current_state, action)
            path.append(next_state)
            
            if next_state == current_state:
                break
            
            current_state = next_state
            steps += 1
        
        return path
    
    def get_reachable_states(self):
        """Get all states from which the goal is reachable by the arm."""
        reachable = []
        for state in range(self.nS):
            if state not in self.obstacles and self.can_reach_goal_from_state(state):
                reachable.append(state)
        return reachable


# ==========================================================================
#                  DYNAMIC PROGRAMMING ALGORITHMS (Unchanged)
# ==========================================================================

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """Evaluate a policy using iterative policy evaluation."""
    V = np.zeros(env.nS)
    
    iteration = 0
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
        
        iteration += 1
        if delta < theta:
            break
    
    return V


def q_from_v(env, V, s, gamma=0.99):
    """Compute Q(s,a) from V(s) for a given state."""
    Q = np.zeros(env.nA)
    
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            if done:
                Q[a] += prob * reward
            else:
                Q[a] += prob * (reward + gamma * V[next_state])
    
    return Q


def policy_improvement(env, V, gamma=0.99):
    """Improve policy by making it greedy w.r.t. value function."""
    policy = np.zeros(env.nS, dtype=int)
    
    for s in range(env.nS):
        Q = q_from_v(env, V, s, gamma)
        policy[s] = np.argmax(Q)
    
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    """Find optimal policy using Policy Iteration."""
    policy = np.zeros(env.nS, dtype=int)
    
    iterations = 0
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        
        iterations += 1
        
        if np.array_equal(policy, new_policy):
            break
        
        policy = new_policy
    
    return policy, V, iterations


def value_iteration(env, gamma=0.99, theta=1e-8):
    """Find optimal policy using Value Iteration."""
    V = np.zeros(env.nS)
    
    iterations = 0
    while True:
        delta = 0
        
        for s in range(env.nS):
            v = V[s]
            Q = q_from_v(env, V, s, gamma)
            V[s] = np.max(Q)
            delta = max(delta, abs(v - V[s]))
        
        iterations += 1
        
        if delta < theta:
            break
    
    policy = policy_improvement(env, V, gamma)
    
    return policy, V, iterations