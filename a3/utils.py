"""
utils.py — Shared utilities for Assignment 3: Biped 1 m Platform Jump.

Contains
--------
  - SAC_CONFIG          Hyperparameters for Soft Actor-Critic (YOU will tune these)
  - Training constants  TOTAL_TIMESTEPS, EVAL_FREQ, EVAL_EPISODES, ROBOT_MASS_KG
  - RewardPlotCallback  Records episode rewards and saves a plot after training
  - BipedJumpEnv        Gymnasium environment — provided, do not modify
"""

# ===========================================================================
# Hyperparameters  (edit these for Task 3)
# ===========================================================================

TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 10_000
MAX_EPISODE_STEPS = 500

# ---------------------------------------------------------------------------
# SAC  (Soft Actor-Critic) — the only algorithm used in this assignment
# ---------------------------------------------------------------------------
SAC_CONFIG = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    buffer_size   = 1_000_000,
    batch_size    = 256,
    tau           = 0.005,
    gamma         = 0.99,
    ent_coef      = "auto",
    verbose       = 1,
)

# ---------------------------------------------------------------------------
# Evaluation / metric settings  (do not change)
# ---------------------------------------------------------------------------
EVAL_EPISODES = 10
ROBOT_MASS_KG = 2.05   # used to compute Cost of Transport (CoT)


# ===========================================================================
# RewardPlotCallback
# ===========================================================================

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless training
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardPlotCallback(BaseCallback):
    """Records episode rewards during training and saves a plot at the end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones",   [False])[0]

        self._current_episode_reward += reward
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True   # returning False would stop training

    def plot_rewards(self, save_path="reward_curve_sac.png"):
        if not self.episode_rewards:
            print("No episode rewards recorded yet.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.6, label="Episode Reward")

        window = 20
        if len(self.episode_rewards) >= window:
            rolling = [
                sum(self.episode_rewards[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.episode_rewards) + 1)
            ]
            plt.plot(rolling, color="red", linewidth=2, label=f"{window}-ep Rolling Avg")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SAC Training Reward Curve — Biped 1 m Jump")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Reward plot saved to {save_path}")


# ===========================================================================
# BipedJumpEnv  — provided environment, do not modify
# ===========================================================================

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

_ASSEST_DIR = os.path.join(os.path.dirname(__file__), "assest")


class BipedJumpEnv(gym.Env):
    """
    Task: the biped robot spawns on top of a 1 m tall platform and must
    jump off, then land upright on the ground below.

    Phases
    ------
    1. On platform  
    2. In flight    
    3. Landing      

   
    """

    PLATFORM_H = 1.0          # top surface height (m)
    SPAWN_Z    = 1.0 + 0.81   # robot COM at spawn  (platform top + standing height)
    GROUND_Z   = 0.81         # robot COM when standing on flat ground

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        cid = p.connect(p.GUI if render else p.DIRECT)
        self.physics_client = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        self.timestep = 1.0 / 50.0
        p.setTimeStep(self.timestep, physicsClientId=cid)

        self.max_steps         = 500
        self.step_counter      = 0
        self.land_stable_steps = 0

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=cid)

        # 1 m platform  (box 1.2 × 1.2 × 1.0 m, centre at z = 0.5)
        plat_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.5],
                                          physicsClientId=cid)
        plat_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.5],
                                       rgbaColor=[0.55, 0.27, 0.07, 1],
                                       physicsClientId=cid)
        self.platform_id = p.createMultiBody(0, plat_col, plat_vis,
                                              [0, 0, 0.5], physicsClientId=cid)

        # Robot
        urdf_path = os.path.join(_ASSEST_DIR, "biped_.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, self.SPAWN_Z],
                                    useFixedBase=False, physicsClientId=cid)
        p.changeDynamics(self.robot_id, -1,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=cid)

        # Joint discovery
        self.joint_indices   = []
        self.joint_limits    = []
        self.left_foot_link  = 2
        self.right_foot_link = 5

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=cid)):
            ji = p.getJointInfo(self.robot_id, i, physicsClientId=cid)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits.append((ji[8], ji[9]))
            if b"left_foot"  in ji[12]: self.left_foot_link  = i
            if b"right_foot" in ji[12]: self.right_foot_link = i

        p.changeDynamics(self.robot_id, self.left_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)
        p.changeDynamics(self.robot_id, self.right_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)

        self.n_actuated = len(self.joint_indices)

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self.n_actuated,), dtype=np.float32)
        obs_dim  = self.n_actuated * 2 + 3 + 3 + 3 + 2 + 1 + 1
        obs_high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.prev_z     = self.SPAWN_Z
        self.has_landed = False
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cid = self.physics_client

        # Reset base pose
        p.resetBasePositionAndOrientation(
            self.robot_id, [0, 0, self.SPAWN_Z],
            p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=cid)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=cid)

        # Reset all joints
        for idx in self.joint_indices:
            p.resetJointState(self.robot_id, idx, 0.0, 0.0,
                              physicsClientId=cid)

        # Reset counters
        self.step_counter = 0
        self.land_stable_steps = 0
        self.prev_z = self.SPAWN_Z
        self.has_landed = False
        self._initial_pos = [0, 0, self.SPAWN_Z]

        # Let the simulation settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=cid)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        cid = self.physics_client

        # Joint positions and velocities
        joint_states = [p.getJointState(self.robot_id, j, physicsClientId=cid)
                        for j in self.joint_indices]
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]

        # Base position, orientation, velocities
        pos, quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=cid)
        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=cid)
        euler = p.getEulerFromQuaternion(quat)

        # Foot contacts with ground plane
        left_contact = len(p.getContactPoints(
            self.robot_id, self.plane_id, self.left_foot_link, -1,
            physicsClientId=cid)) > 0
        right_contact = len(p.getContactPoints(
            self.robot_id, self.plane_id, self.right_foot_link, -1,
            physicsClientId=cid)) > 0

        obs = np.array(
            joint_pos + joint_vel +
            list(pos) + list(lin_vel) + list(euler) +
            [float(left_contact), float(right_contact)] +
            [pos[2]] +
            [float(self.has_landed)],
            dtype=np.float32
        )
        return obs

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, orn, lin_vel, landed_now, on_platform):
        euler = p.getEulerFromQuaternion(orn)
        tilt = abs(euler[0]) + abs(euler[1])

        reward = 0.0

        # ── Always-on: upright bonus ──
        # Smooth 0→1 signal: 1.0 when perfectly vertical, 0 when tilt ≥ 1.0
        upright = max(0.0, 1.0 - tilt)
        reward += upright * 1.0

        # ── Phase 1: On Platform — get off ASAP ──
        if on_platform:
            reward -= 0.5                              # mild tick penalty — time to prepare
            reward += max(lin_vel[0], 0.0) * 1.0       # nudge forward

        # ── Phase 2: In Flight — descend upright ──
        elif not self.has_landed:
            # Dense height signal: reward getting closer to ground
            height_above = max(pos[2] - self.GROUND_Z, 0.0)
            reward -= height_above * 0.5               # continuous pull down
            reward += (self.prev_z - pos[2]) * 5.0     # delta reward for descending
            # Extra bonus for controlled flight
            if tilt < 0.5:
                reward += 2.0

        # ── Phase 3: Post-Landing — stay upright ──
        else:
            if tilt < 0.3:
                reward += 3.0                          # stability bonus per step
            else:
                reward -= 1.0                          # penalize wobble

        # ── One-time landing bonus (graded by uprightness) ──
        if landed_now:
            if tilt < 0.3:
                reward += 200.0
            elif tilt < 0.8:
                reward += 50.0
            else:
                reward += 10.0

        self.prev_z = pos[2]
        return reward

    # ------------------------------------------------------------------
    def get_joint_indices(self):
        return list(self.joint_indices)

    def robot_initial_position(self):
        return list(self._initial_pos)

    def robot_current_position(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client)
        return list(pos)

    # ------------------------------------------------------------------
    def step(self, action):
        cid = self.physics_client
        action = np.clip(action, -1.0, 1.0)

        # Apply actions as velocity targets scaled by joint velocity limits
        for i, idx in enumerate(self.joint_indices):
            lo, hi = self.joint_limits[i]
            max_vel = 2.0  # max velocity from URDF
            target_vel = float(action[i]) * max_vel
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=15.0,
                physicsClientId=cid)

        p.stepSimulation(physicsClientId=cid)
        self.step_counter += 1

        # Get state
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=cid)
        lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=cid)

        # Check foot contacts with ground
        left_contact = len(p.getContactPoints(
            self.robot_id, self.plane_id, self.left_foot_link, -1,
            physicsClientId=cid)) > 0
        right_contact = len(p.getContactPoints(
            self.robot_id, self.plane_id, self.right_foot_link, -1,
            physicsClientId=cid)) > 0
        both_feet = left_contact and right_contact

        # Check foot contacts with platform
        left_plat = len(p.getContactPoints(self.robot_id, self.platform_id, self.left_foot_link, -1, physicsClientId=cid)) > 0
        right_plat = len(p.getContactPoints(self.robot_id, self.platform_id, self.right_foot_link, -1, physicsClientId=cid)) > 0
        on_platform = left_plat or right_plat

        # Landing detection
        landed_now = False
        if both_feet and pos[2] < 1.15 and not self.has_landed:
            self.has_landed = True
            landed_now = True

        # Stable landing counter
        euler = p.getEulerFromQuaternion(orn)
        tilt = abs(euler[0]) + abs(euler[1])
        if self.has_landed and both_feet and tilt < 0.3:
            self.land_stable_steps += 1
        else:
            if self.has_landed:
                self.land_stable_steps = 0

        # Compute reward
        reward = self._compute_reward(pos, orn, lin_vel, landed_now, on_platform)

        # Termination conditions
        terminated = False
        # Crashed: extreme tilt ONLY when actually on the ground floor
        # GROUND_Z=0.81, so z<0.5 means the robot is lying flat on the floor
        if tilt > 1.5 and pos[2] < 0.5:
            terminated = True
            reward -= 50.0
        # Stable landing achieved — success!
        if self.land_stable_steps >= 30:
            terminated = True
            reward += 50.0
        # Fell below ground
        if pos[2] < 0.2:
            terminated = True
            reward -= 50.0

        truncated = self.step_counter >= self.max_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def close(self):
        p.disconnect(self.physics_client)
