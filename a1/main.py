"""
==========================================================================
                    MAIN.PY - UR5 GRID NAVIGATION
==========================================================================
Students implement DP algorithms in utils.py and run this to see results.

Dependencies:
    - pybullet
    - numpy
    - utils.py

Usage:
    python main.py

Author: Assignment 1 - AR525
==========================================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import sys


from utils import (
    GridEnv,
    policy_iteration,
    value_iteration
)



def state_to_position(state, rows, cols, grid_size=0.10, 
                      table_center=[0, -0.3, 0.65], z_offset=0.10):

    row = state // cols
    col = state % cols
    
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    
    return [x, y, z]


def draw_grid_lines(rows, cols, grid_size=0.10, table_center=[0, -0.3, 0.65]):

    line_color = [0, 0, 0]
    line_width = 2
    z = table_center[2] + 0.001
    
    x_start = table_center[0] - (cols/2) * grid_size
    x_end = table_center[0] + (cols/2) * grid_size
    y_start = table_center[1] - (rows/2) * grid_size
    y_end = table_center[1] + (rows/2) * grid_size
    

    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_end, y, z], line_color, line_width)

    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_end, z], line_color, line_width)


def draw_value_heatmap(env, V, grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Draw a heatmap visualization of the value function on the grid.
    
    Args:
        env: GridEnv environment
        V: Value function array
        grid_size: Size of each grid cell
        table_center: Center position of the table
    """
    # Normalize values for color mapping
    V_min = np.min(V)
    V_max = np.max(V)
    if V_max - V_min > 0:
        V_norm = (V - V_min) / (V_max - V_min)
    else:
        V_norm = np.zeros_like(V)
    
    z = table_center[2] + 0.002
    half = grid_size / 2 * 0.85
    
    for s in range(env.nS):
        pos = state_to_position(s, env.rows, env.cols, grid_size, table_center, z_offset=0.002)
        
        # Color gradient: blue (low) -> green -> red (high)
        val = V_norm[s]
        if val < 0.5:
            # Blue to Green
            r, g, b = 0, 2 * val, 1 - 2 * val
        else:
            # Green to Red  
            r, g, b = 2 * (val - 0.5), 1 - 2 * (val - 0.5), 0
        
        color = [r, g, b]
        
        # Draw value as text
        value_text = f"{V[s]:.1f}"
        p.addUserDebugText(value_text, [pos[0], pos[1], pos[2] + 0.03], 
                          textColorRGB=[0, 0, 0], textSize=0.8)


def draw_optimal_path(path, env, grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Draw the optimal path on the grid with arrows.
    
    Args:
        path: List of states representing the optimal path
        env: GridEnv environment
        grid_size: Size of each grid cell
        table_center: Center position of the table
    """
    z = table_center[2] + 0.01
    path_color = [0, 0.8, 0]  # Green
    
    for i in range(len(path) - 1):
        start_pos = state_to_position(path[i], env.rows, env.cols, grid_size, table_center, z_offset=0.01)
        end_pos = state_to_position(path[i+1], env.rows, env.cols, grid_size, table_center, z_offset=0.01)
        
        # Draw path line
        p.addUserDebugLine(start_pos, end_pos, path_color, lineWidth=4)


def move_robot_along_path(ur5_id, path, env, grid_size=0.10, table_center=[0, -0.3, 0.65]):
    """
    Move the UR5 robot along the optimal path using inverse kinematics.
    
    Args:
        ur5_id: PyBullet ID of the UR5 robot
        path: List of states representing the optimal path
        env: GridEnv environment
        grid_size: Size of each grid cell
        table_center: Center position of the table
    """
    end_effector_link = 6  # UR5 end-effector link index
    trail_color = [0, 1, 0]  # Green trail
    prev_pos = None
    
    for state in path:
        target_pos = state_to_position(state, env.rows, env.cols, grid_size, table_center, z_offset=0.10)
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # End-effector pointing down
        
        # Calculate inverse kinematics
        joint_angles = p.calculateInverseKinematics(
            ur5_id, 
            end_effector_link,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Move robot smoothly
        num_steps = 50
        current_angles = [p.getJointState(ur5_id, i)[0] for i in range(6)]
        
        for step in range(num_steps):
            t = (step + 1) / num_steps
            interpolated = [
                current_angles[i] + t * (joint_angles[i] - current_angles[i])
                for i in range(6)
            ]
            
            for i in range(6):
                p.setJointMotorControl2(
                    ur5_id, i, 
                    p.POSITION_CONTROL,
                    targetPosition=interpolated[i],
                    force=500
                )
            
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Get actual end-effector position and draw trail
        ee_state = p.getLinkState(ur5_id, end_effector_link)
        ee_pos = ee_state[0]
        
        if prev_pos is not None:
            p.addUserDebugLine(prev_pos, ee_pos, trail_color, lineWidth=3)
        
        prev_pos = ee_pos
        
        # Pause briefly at each cell
        for _ in range(20):
            p.stepSimulation()
            time.sleep(1./240.)


def print_policy_visualization(env, policy):
    """
    Print a text visualization of the policy.
    
    Args:
        env: GridEnv environment
        policy: Policy array
    """
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    print("\n=== Optimal Policy ===")
    for row in range(env.rows):
        row_str = ""
        for col in range(env.cols):
            state = row * env.cols + col
            if state == env.start:
                row_str += " S "
            elif state == env.goal:
                row_str += " G "
            else:
                row_str += f" {arrows[policy[state]]} "
        print(row_str)
    print()


def print_value_function(env, V):
    """
    Print a text visualization of the value function.
    
    Args:
        env: GridEnv environment
        V: Value function array
    """
    print("\n=== Value Function ===")
    for row in range(env.rows):
        row_str = ""
        for col in range(env.cols):
            state = row * env.cols + col
            row_str += f"{V[state]:7.2f} "
        print(row_str)
    print()




if __name__ == "__main__":
    

    ROWS = 5
    COLS = 6
    START = 0
    GOAL = ROWS * COLS - 1
    GAMMA = 0.99
    THETA = 1e-8
    
    # ==========================================================================
    # Generate obstacles BEFORE creating environment (for Part 6 compatibility)
    # ==========================================================================
    all_states = set(range(ROWS * COLS))
    available_states = list(all_states - {START, GOAL})
    
    num_obstacles = min(5, len(available_states))
    np.random.seed(42)  # For reproducibility (can be removed for random testing)
    obstacle_states = list(np.random.choice(available_states, num_obstacles, replace=False))
    
    print(f"Grid: {ROWS}x{COLS}, Start: {START}, Goal: {GOAL}")
    print(f"Obstacles at states: {obstacle_states}")
    
    # Create environment WITH obstacles
    env = GridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL, obstacles=obstacle_states)

    # ==========================================================================
    # Run Dynamic Programming Algorithms
    # ==========================================================================
    print("=" * 60)
    print("    AR525 Assignment 1: Grid Navigation using DP")
    print("=" * 60)
    
    # Policy Iteration
    print("\n>>> Running Policy Iteration...")
    start_time = time.time()
    pi_policy, pi_V, pi_iters = policy_iteration(env, gamma=GAMMA, theta=THETA)
    pi_time = time.time() - start_time
    print(f"Policy Iteration completed in {pi_iters} iterations ({pi_time:.4f}s)")
    
    # Value Iteration
    print("\n>>> Running Value Iteration...")
    start_time = time.time()
    vi_policy, vi_V, vi_iters = value_iteration(env, gamma=GAMMA, theta=THETA)
    vi_time = time.time() - start_time
    print(f"Value Iteration completed in {vi_iters} iterations ({vi_time:.4f}s)")
    
    # Compare results
    print("\n=== Algorithm Comparison ===")
    print(f"Policy Iteration: {pi_iters} iterations, {pi_time:.4f}s")
    print(f"Value Iteration:  {vi_iters} iterations, {vi_time:.4f}s")
    print(f"Policies match: {np.array_equal(pi_policy, vi_policy)}")
    
    # Use policy iteration result for visualization
    optimal_policy = pi_policy
    optimal_V = pi_V
    
    # Print policy and value function
    print_policy_visualization(env, optimal_policy)
    print_value_function(env, optimal_V)
    
    # Extract optimal path
    optimal_path = env.get_optimal_path(optimal_policy)
    print(f"Optimal path length: {len(optimal_path)} states")
    print(f"Optimal path: {optimal_path}")
    
    # ==========================================================================
    # PyBullet Simulation Setup
    # ==========================================================================

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    p.loadURDF("plane.urdf")
    
    table_path = os.path.join("assest", "table", "table.urdf")
    p.loadURDF(table_path, [0, -0.3, 0], globalScaling=2.0)
    
    stand_path = os.path.join("assest", "robot_stand.urdf")
    p.loadURDF(stand_path, [0, -0.8, 0], useFixedBase=True)
    
    ur5_path = os.path.join("assest", "ur5.urdf")
    ur5_start_pos = [0, -0.8, 0.65]
    ur5_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    ur5_id = p.loadURDF(ur5_path, ur5_start_pos, ur5_start_orn, useFixedBase=True)
    
    sys.stderr = old_stderr
    
    draw_grid_lines(env.rows, env.cols)
    
    # Place obstacle visuals (using obstacles already defined in env)
    obstacle_path = os.path.join("assest", "cube_and_square", "cube_small_cyan.urdf")
    for obs_state in obstacle_states:
        obs_pos = state_to_position(obs_state, env.rows, env.cols, z_offset=0.025)
        p.loadURDF(obstacle_path, obs_pos)

    grid_size = 0.10
    half = grid_size / 2 * 0.8

    # Draw start marker (yellow)
    start_pos = state_to_position(env.start, env.rows, env.cols, z_offset=0.005)
    yellow = [1, 1, 0]
    p.addUserDebugLine([start_pos[0]-half, start_pos[1]-half, start_pos[2]], 
                       [start_pos[0]+half, start_pos[1]-half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]+half, start_pos[1]-half, start_pos[2]], 
                       [start_pos[0]+half, start_pos[1]+half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]+half, start_pos[1]+half, start_pos[2]], 
                       [start_pos[0]-half, start_pos[1]+half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]-half, start_pos[1]+half, start_pos[2]], 
                       [start_pos[0]-half, start_pos[1]-half, start_pos[2]], yellow, 3, 0)
    
    # Draw goal marker (red)
    goal_pos = state_to_position(env.goal, env.rows, env.cols, z_offset=0.005)
    red = [1, 0, 0]
    p.addUserDebugLine([goal_pos[0]-half, goal_pos[1]-half, goal_pos[2]], 
                       [goal_pos[0]+half, goal_pos[1]-half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]+half, goal_pos[1]-half, goal_pos[2]], 
                       [goal_pos[0]+half, goal_pos[1]+half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]+half, goal_pos[1]+half, goal_pos[2]], 
                       [goal_pos[0]-half, goal_pos[1]+half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]-half, goal_pos[1]+half, goal_pos[2]], 
                       [goal_pos[0]-half, goal_pos[1]-half, goal_pos[2]], red, 3, 0)
    
    # ==========================================================================
    # Visualizations
    # ==========================================================================
    
    # Draw value function heatmap
    draw_value_heatmap(env, optimal_V)
    
    # Draw optimal path on grid
    draw_optimal_path(optimal_path, env)
    
    # Pause before robot movement
    print("\n>>> Starting robot movement in 2 seconds...")
    time.sleep(2)
    
    # Move robot along optimal path
    print(">>> Moving robot along optimal path...")
    move_robot_along_path(ur5_id, optimal_path, env)
    
    print("\n>>> Robot reached the goal! Simulation complete.")
    print(">>> Press Ctrl+C to exit.")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nSimulation terminated.")
        p.disconnect()
