"""
==========================================================================
          MAIN.PY - MOBILE MANIPULATOR WITH ARM REACH
==========================================================================
The end-effector must reach the goal. The mobile base moves to positions
where the arm can reach the goal. The arm then extends to the goal.

Dependencies:
    - pybullet
    - numpy
    - matplotlib
    - utils_mobile.py

Usage:
    python main_mobile.py --rows 10 --cols 12 --arm_reach 2.5

Author: Assignment 1 - AR525 (Mobile Manipulator Extension)
==========================================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse

from utils_mobile import (
    MobileManipulatorEnv,
    policy_iteration,
    value_iteration
)


def state_to_position(state, rows, cols, grid_size=0.20, floor_center=[0, 0, 0]):
    """Convert grid state to 3D position on the floor."""
    row = state // cols
    col = state % cols
    
    x = floor_center[0] + (col - cols/2 + 0.5) * grid_size
    y = floor_center[1] + (row - rows/2 + 0.5) * grid_size
    z = floor_center[2]
    
    return [x, y, z]


def draw_grid_lines(rows, cols, grid_size=0.20, floor_center=[0, 0, 0]):
    """Draw grid lines on the floor."""
    line_color = [0.3, 0.3, 0.3]
    line_width = 2
    z = floor_center[2] + 0.01
    
    x_start = floor_center[0] - (cols/2) * grid_size
    x_end = floor_center[0] + (cols/2) * grid_size
    y_start = floor_center[1] - (rows/2) * grid_size
    y_end = floor_center[1] + (rows/2) * grid_size
    
    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_end, y, z], line_color, line_width)
    
    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_end, z], line_color, line_width)


def draw_arm_reach_zone(env, state, grid_size=0.20, floor_center=[0, 0, 0]):
    """Draw a circle showing the arm's reach from a given base position."""
    base_pos = state_to_position(state, env.rows, env.cols, grid_size, floor_center)
    base_pos[2] = 0.02
    
    # Draw reach circle
    num_segments = 32
    reach_radius = env.arm_reach * grid_size
    color = [0, 0.7, 0.7, 0.3]  # Cyan, semi-transparent
    
    for i in range(num_segments):
        angle1 = 2 * np.pi * i / num_segments
        angle2 = 2 * np.pi * (i + 1) / num_segments
        
        p1 = [base_pos[0] + reach_radius * np.cos(angle1),
              base_pos[1] + reach_radius * np.sin(angle1),
              base_pos[2]]
        p2 = [base_pos[0] + reach_radius * np.cos(angle2),
              base_pos[1] + reach_radius * np.sin(angle2),
              base_pos[2]]
        
        p.addUserDebugLine(p1, p2, color[:3], lineWidth=2)


def draw_reachable_zone(env, grid_size=0.20, floor_center=[0, 0, 0]):
    """Highlight all grid cells from which the goal is reachable."""
    reachable_states = env.get_reachable_states()
    
    z = floor_center[2] + 0.015
    half = grid_size / 2 * 0.9
    color = [0, 0.8, 0.8]  # Cyan
    
    for state in reachable_states:
        pos = state_to_position(state, env.rows, env.cols, grid_size, floor_center)
        pos[2] = z
        
        # Draw filled square
        p.addUserDebugLine(
            [pos[0]-half, pos[1]-half, pos[2]], 
            [pos[0]+half, pos[1]-half, pos[2]], 
            color, lineWidth=3
        )
        p.addUserDebugLine(
            [pos[0]+half, pos[1]-half, pos[2]], 
            [pos[0]+half, pos[1]+half, pos[2]], 
            color, lineWidth=3
        )
        p.addUserDebugLine(
            [pos[0]+half, pos[1]+half, pos[2]], 
            [pos[0]-half, pos[1]+half, pos[2]], 
            color, lineWidth=3
        )
        p.addUserDebugLine(
            [pos[0]-half, pos[1]+half, pos[2]], 
            [pos[0]-half, pos[1]-half, pos[2]], 
            color, lineWidth=3
        )


def draw_optimal_path(path, env, grid_size=0.20, floor_center=[0, 0, 0]):
    """Draw the optimal path for the mobile base."""
    z = floor_center[2] + 0.025
    path_color = [0, 0.8, 0]
    
    for i in range(len(path) - 1):
        start_pos = state_to_position(path[i], env.rows, env.cols, grid_size, floor_center)
        end_pos = state_to_position(path[i+1], env.rows, env.cols, grid_size, floor_center)
        
        start_pos[2] = z
        end_pos[2] = z
        
        p.addUserDebugLine(start_pos, end_pos, path_color, lineWidth=5)


def get_joint_indices(robot_id, joint_names):
    ids = []
    for i in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, i)[1].decode("utf-8")
        if name in joint_names:
            ids.append(i)
    return ids

def extend_arm_to_goal(robot_id, base_state, goal_state, env, grid_size=0.20, 
                       floor_center=[0, 0, 0]):
    """
    Extend arm to goal using proper IK.
    """
    base_pos = state_to_position(base_state, env.rows, env.cols, grid_size, floor_center)
    goal_pos = state_to_position(goal_state, env.rows, env.cols, grid_size, floor_center)
    
    # Calculate distance
    dx = goal_pos[0] - base_pos[0]
    dy = goal_pos[1] - base_pos[1]
    dist_2d = np.sqrt(dx**2 + dy**2)
    
    print(f"\n>>> Extending arm to goal...")
    print(f"    Base: [{base_pos[0]:.2f}, {base_pos[1]:.2f}]")
    print(f"    Goal: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}]")
    print(f"    Distance: {dist_2d:.2f}m ({dist_2d/grid_size:.2f} cells)")
    
    # Target position (slightly above goal)
    target_pos = [goal_pos[0], goal_pos[1], 0.5]  # 15cm above ground
    target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])  # Point down
    
    # UR5 arm joints
    arm_joints = [6, 7, 8, 9, 10, 11]  # shoulder_pan to wrist_3
    ee_link_index = 13  # End-effector link
    
    # Get current angles
    current_angles = [p.getJointState(robot_id, j)[0] for j in arm_joints]
    print(f"    Current: {[f'{np.degrees(a):.1f}°' for a in current_angles]}")
    
    # Calculate IK with proper parameters
    ik_result = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link_index,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        lowerLimits=[-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi],
        upperLimits=[2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi],
        jointRanges=[4*np.pi, 4*np.pi, 2*np.pi, 4*np.pi, 4*np.pi, 4*np.pi],
        restPoses=current_angles,
        maxNumIterations=200,
        residualThreshold=1e-5
    )
    
    target_angles = list(ik_result[:6])
    print(f"    Target:  {[f'{np.degrees(a):.1f}°' for a in target_angles]}")
    
    # Smooth interpolation
    num_steps = 120
    prev_ee_pos = None
    
    for step in range(num_steps + 1):
        t = step / num_steps
        
        # Interpolate
        interp_angles = [current_angles[i] + t * (target_angles[i] - current_angles[i]) 
                        for i in range(6)]
        
        # Apply to joints
        for i, joint_idx in enumerate(arm_joints):
            p.setJointMotorControl2(
                robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=interp_angles[i],
                force=500,
                maxVelocity=2.0
            )
        
        p.stepSimulation()
        time.sleep(1./240.)
        
        # Draw trail
        if step % 3 == 0:
            ee_state = p.getLinkState(robot_id, ee_link_index)
            ee_pos = list(ee_state[0])
            
            if prev_ee_pos:
                p.addUserDebugLine(prev_ee_pos, ee_pos, [1, 0.5, 0], 5, 0)
            prev_ee_pos = ee_pos
    # At the end of extend_arm_to_goal function, add:

    # Check final position
    final_ee = p.getLinkState(robot_id, ee_link_index)[0]
    error = np.sqrt(sum((final_ee[i] - target_pos[i])**2 for i in range(3)))
    
    print(f"    Final EE: [{final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f}]")
    print(f"    Error: {error*100:.2f}cm")
    
    # Draw line to goal
    p.addUserDebugLine(final_ee, [goal_pos[0], goal_pos[1], 0.02], [1,0,0], 8, 0)
    
    # Success marker
    marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1,0,0,1])
    p.createMultiBody(0, baseVisualShapeIndex=marker, 
                     basePosition=[goal_pos[0], goal_pos[1], 0.08])
    

    # Pause
    for _ in range(80):
        p.stepSimulation()
        time.sleep(1./240.)

def show_value_heatmap(V, path, rows, cols, title="Value Function Heatmap"):
    """
    Display a optimal path and 2D heatmap of the value function.

    Args:
        V: value function (1D array of size rows*cols)
        rows: number of grid rows
        cols: number of grid columns
    """
    V_grid = V.reshape(rows, cols)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Heatmap
    im = ax.imshow(V_grid, cmap="jet", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    mean_val = np.mean(V_grid)
    for r in range(rows):
        for c in range(cols):
            ax.text(
                c, r, f"{V_grid[r, c]:.1f}",ha="center", va="center",
                color="white", fontsize=11
            )

    # ---- Draw optimal path ----
    if path is not None and len(path) > 1:
        path_rc = [(s // cols, s % cols) for s in path]

        ys = [r for r, c in path_rc]
        xs = [c for r, c in path_rc]

        # Path line
        ax.plot(xs, ys, color="lime", linewidth=2, label="Optimal Path")

        # Direction arrows
        for i in range(len(xs) - 1):
            ax.arrow(
                xs[i], ys[i],xs[i+1] - xs[i], ys[i+1] - ys[i],
                head_width=0.15, head_length=0.15,
                fc="lime", ec="lime",length_includes_head=True
            )

        ax.legend(loc="lower right")

    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def move_mobile_base_along_path(robot_id, path, env, grid_size=0.20, floor_center=[0, 0, 0]):
    """Move the mobile base along the optimal path."""
    trail_color = [0, 1, 0]
    prev_pos = None
    
    for state in path:
        target_pos = state_to_position(state, env.rows, env.cols, grid_size, floor_center)
        
        current_pos, current_orn = p.getBasePositionAndOrientation(robot_id)
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            target_yaw = np.arctan2(dy, dx)
            target_orn = p.getQuaternionFromEuler([0, 0, target_yaw])
        else:
            target_orn = current_orn
        
        # Smooth movement
        num_steps = 60
        
        for step in range(num_steps):
            t = (step + 1) / num_steps
            
            interp_pos = [
                current_pos[0] + t * dx,
                current_pos[1] + t * dy,
                0.0
            ]
            
            interp_orn = p.getQuaternionFromEuler([
                0, 0, 
                p.getEulerFromQuaternion(current_orn)[2] + 
                t * (p.getEulerFromQuaternion(target_orn)[2] - 
                     p.getEulerFromQuaternion(current_orn)[2])
            ])
            
            p.resetBasePositionAndOrientation(robot_id, interp_pos, interp_orn)
            
            p.stepSimulation()
            time.sleep(1./240.)
        
        # Draw trail
        base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        trail_pos = [base_pos[0], base_pos[1], 0.01]
        
        if prev_pos is not None:
            p.addUserDebugLine(prev_pos, trail_pos, trail_color, lineWidth=6)
        
        prev_pos = trail_pos
        
        # Draw reach zone at this position
        draw_arm_reach_zone(env, state, grid_size, floor_center)
        
        for _ in range(15):
            p.stepSimulation()
            time.sleep(1./240.)


def print_policy_visualization(env, policy):
    """Print text visualization of the policy."""
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    print("\n=== Optimal Policy (Mobile Base Movement) ===")
    for row in range(env.rows):
        row_str = ""
        for col in range(env.cols):
            state = row * env.cols + col
            if state == env.start:
                row_str += " S "
            elif state == env.goal:
                row_str += " G "
            elif state in env.obstacles:
                row_str += " X "
            elif env.can_reach_goal_from_state(state):
                row_str += " ✓ "  # Goal reachable from here
            else:
                row_str += f" {arrows[policy[state]]} "
        print(row_str)
    print()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="AR525: Mobile Manipulator - End-Effector Reaches Goal"
    )
    parser.add_argument("--rows", type=int, default=10, help="Grid rows")
    parser.add_argument("--cols", type=int, default=12, help="Grid columns")
    parser.add_argument("--arm_reach", type=float, default=2.5, 
                       help="Arm reach in grid cells (default: 2.5)")
    args = parser.parse_args()

    ROWS = args.rows
    COLS = args.cols
    ARM_REACH = args.arm_reach  # in grid cells
    START = 0
    GOAL = ROWS * COLS - 1
    GAMMA = 0.99
    THETA = 1e-8
    GRID_SIZE = 0.4
    
    # ==========================================================================
    # Generate obstacles (ensuring start and goal neighbors are clear)
    # ==========================================================================
    all_states = set(range(ROWS * COLS))
    
    reserved = {START, GOAL}
    # Add neighbors of start
    if START % COLS > 0:
        reserved.add(START - 1)
    if START % COLS < COLS - 1:
        reserved.add(START + 1)
    if START >= COLS:
        reserved.add(START - COLS)
    if START < ROWS * COLS - COLS:
        reserved.add(START + COLS)
    
    # Add neighbors of goal
    if GOAL % COLS > 0:
        reserved.add(GOAL - 1)
    if GOAL % COLS < COLS - 1:
        reserved.add(GOAL + 1)
    if GOAL >= COLS:
        reserved.add(GOAL - COLS)
    if GOAL < ROWS * COLS - COLS:
        reserved.add(GOAL + COLS)
    
    available_states = list(all_states - reserved)
    num_obstacles = min(int(ROWS * COLS * 0.15), len(available_states))
    np.random.seed(23)
    obstacle_states = list(np.random.choice(available_states, num_obstacles, replace=False))
    
    print(f"\n{'='*75}")
    print(f"  AR525: Mobile Manipulator - End-Effector Goal Reaching")
    print(f"{'='*75}")
    print(f"Grid: {ROWS}×{COLS} ({ROWS*COLS} states)")
    print(f"Start (base): {START}, Goal (EE target): {GOAL}")
    print(f"Arm reach: {ARM_REACH} cells ({ARM_REACH*GRID_SIZE:.2f}m)")
    print(f"Obstacles: {len(obstacle_states)} cells")
    
    # Create environment
    env = MobileManipulatorEnv(
        rows=ROWS, cols=COLS, start=START, goal=GOAL, 
        obstacles=obstacle_states, arm_reach=ARM_REACH, grid_size=GRID_SIZE
    )
    
    # Check if goal is directly reachable from start
    reachable_states = env.get_reachable_states()
    print(f"States where goal is reachable: {len(reachable_states)}")
    
    if env.can_reach_goal_from_state(START):
        print(f"✓ Goal is directly reachable from start! (distance: {env._distance(START, GOAL):.2f} cells)")
    else:
        print(f"✗ Goal NOT directly reachable from start (distance: {env._distance(START, GOAL):.2f} cells)")
        print(f"  Mobile base must move to reach zone first")
    
    # ==========================================================================
    # Run DP Algorithms
    # ==========================================================================
    
    print(f"\n{'='*75}")
    print(">>> Running Policy Iteration...")
    start_time = time.time()
    pi_policy, pi_V, pi_iters = policy_iteration(env, gamma=GAMMA, theta=THETA)
    pi_time = time.time() - start_time
    print(f"✓ Completed in {pi_iters} iterations ({pi_time:.4f}s)")
    
    print("\n>>> Running Value Iteration...")
    start_time = time.time()
    vi_policy, vi_V, vi_iters = value_iteration(env, gamma=GAMMA, theta=THETA)
    vi_time = time.time() - start_time
    print(f"✓ Completed in {vi_iters} iterations ({vi_time:.4f}s)")
    
    print(f"\n{'='*75}")
    print("=== Algorithm Comparison ===")
    print(f"Policy Iteration: {pi_iters} iterations, {pi_time:.4f}s")
    print(f"Value Iteration:  {vi_iters} iterations, {vi_time:.4f}s")
    print(f"Policies match: {np.array_equal(pi_policy, vi_policy)}")
    
    optimal_policy = pi_policy
    optimal_V = pi_V
    
    print_policy_visualization(env, optimal_policy)
    
    # Extract optimal path
    optimal_path = env.get_optimal_path(optimal_policy)
    final_base_state = optimal_path[-1]
    
    print(f"\n{'='*75}")
    print(f"Optimal base path length: {len(optimal_path)} states")
    print(f"Path: {optimal_path[:10]}{'...' if len(optimal_path) > 10 else ''}")
    print(f"Final base position: state {final_base_state}")
    print(f"Distance from final base to goal: {env._distance(final_base_state, GOAL):.2f} cells")
    print(f"Goal reachable: {env.can_reach_goal_from_state(final_base_state)}")
    print(f"{'='*75}\n")
    
    # ==========================================================================
    # PyBullet Simulation
    # ==========================================================================
    
    print(">>> Initializing PyBullet simulation...")
    
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    camera_distance = max(ROWS, COLS) * GRID_SIZE * 1.5
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )
    
    p.loadURDF("plane.urdf")

    # Load mobile robot
    mobile_ur5_path = "../assest/mobile_ur5.urdf"
    start_pos = state_to_position(env.start, env.rows, env.cols, GRID_SIZE)
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    
    robot_id = p.loadURDF(mobile_ur5_path, start_pos, start_orn)
    home_poses = [0, -1.57, 1.57, -1.57, 1.57, 0]
    arm_joint_indices = [6, 7, 8, 9, 10, 11]

    for idx, angle in zip(arm_joint_indices, home_poses):
        p.resetJointState(robot_id, idx, angle)
        
    sys.stderr = old_stderr
    
    print("✓ Simulation initialized")
    
    # Draw grid and visualization
    draw_grid_lines(env.rows, env.cols, GRID_SIZE)
    draw_reachable_zone(env, GRID_SIZE)  # Show where goal is reachable
    
    # Place obstacles
    for obs_state in obstacle_states:
        obs_pos = state_to_position(obs_state, env.rows, env.cols, GRID_SIZE)
        obs_pos[2] = 0.05
        
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[GRID_SIZE*0.4, GRID_SIZE*0.4, 0.05],
            rgbaColor=[0, 0.8, 0.8, 1]
        )
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[GRID_SIZE*0.4, GRID_SIZE*0.4, 0.05]
        )
        p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=obs_pos
        )
    
    draw_optimal_path(optimal_path, env, GRID_SIZE)
    show_value_heatmap(optimal_V, optimal_path,env.rows, env.cols)
    
    # Mark start and goal
    half = GRID_SIZE / 2 * 0.8
    
    # Start marker (yellow)
    start_vis_pos = state_to_position(env.start, env.rows, env.cols, GRID_SIZE)
    start_vis_pos[2] = 0.02
    yellow = [1, 1, 0]
    for i in range(4):
        p.addUserDebugLine(
            [start_vis_pos[0] + [-half, half, half, -half][i],
             start_vis_pos[1] + [-half, -half, half, half][i],
             start_vis_pos[2]],
            [start_vis_pos[0] + [half, half, -half, -half][i],
             start_vis_pos[1] + [-half, half, half, -half][i],
             start_vis_pos[2]],
            yellow, 4, 0
        )
    
    # Goal marker (red)
    goal_vis_pos = state_to_position(env.goal, env.rows, env.cols, GRID_SIZE)
    goal_vis_pos[2] = 0.02
    red = [1, 0, 0]
    for i in range(4):
        p.addUserDebugLine(
            [goal_vis_pos[0] + [-half, half, half, -half][i],
             goal_vis_pos[1] + [-half, -half, half, half][i],
             goal_vis_pos[2]],
            [goal_vis_pos[0] + [half, half, -half, -half][i],
             goal_vis_pos[1] + [-half, half, half, -half][i],
             goal_vis_pos[2]],
            red, 4, 0
        )
    
    # ==========================================================================
    # Execute Movement
    # ==========================================================================
    
    print(f"\n{'='*75}")
    print(">>> Starting mobile manipulator motion in 3 seconds...")
    print(f"{'='*75}\n")
    time.sleep(3)
    
    # Move base to optimal position
    print(">>> Phase 1: Moving mobile base to reach zone...")
    move_mobile_base_along_path(robot_id, optimal_path, env, GRID_SIZE)
    
    print(f"\n{'='*75}")
    print(">>> Phase 2: Extending arm to goal...")
    print(f"{'='*75}")
    
    # Extend arm to goal
    extend_arm_to_goal(robot_id, final_base_state, env.goal, env, GRID_SIZE)
    
    print(f"\n{'='*75}")
    print(">>> SUCCESS! End-effector reached the goal!")
    print(">>> Press Ctrl+C to exit.")
    print(f"{'='*75}\n")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\n\nSimulation terminated by user.")
        p.disconnect()