"""
main.py — Assignment 3: Biped RL (1 m Platform Jump with SAC)

Usage examples
--------------
# View the environment (biped + stair in GUI, no model needed):
    python main.py --mode view

# Train SAC (timesteps set in utils.py):
    python main.py --mode train

# Train SAC for a custom number of steps:
    python main.py --mode train --timesteps 500000

# Evaluate the best saved checkpoint (10 episodes, headless):
    python main.py --mode test

# Evaluate with GUI rendering:
    python main.py --mode test --render --episodes 5

# Evaluate a specific model file:
    python main.py --mode test --model_path "models/sac_best/best_model"
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from utils import (
    BipedJumpEnv, RewardPlotCallback,
    TOTAL_TIMESTEPS, EVAL_FREQ,
    SAC_CONFIG,
    EVAL_EPISODES, ROBOT_MASS_KG,
)

# ── Algorithm registry ────────────────────────────────────────────────────────
ALGO_MAP = {
    "sac": (SAC, SAC_CONFIG),
}


# ── Checkpoint Callback ────────────────────────────────────────────────────────
class CheckpointCallback(BaseCallback):
    """
    Saves a full model checkpoint (weights + replay buffer) every
    `save_freq` training steps to `save_path/step_{n_calls}.zip`.
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # Clean up old checkpoints in this dir
            for old_file in glob.glob(os.path.join(self.save_path, "step_*.zip")):
                os.remove(old_file)

            ckpt = os.path.join(self.save_path, f"step_{self.num_timesteps}")
            self.model.save(ckpt)
            if self.verbose:
                print(f"  [checkpoint] saved {ckpt}.zip")
        return True


def _find_latest_checkpoint(ckpt_dir: str):
    """Return (model_path, buffer_path, step_number) for the most recent checkpoint."""
    pattern = os.path.join(ckpt_dir, "step_*.zip")
    files = glob.glob(pattern)
    if not files:
        return None, None, 0
    # Extract step numbers and find the maximum
    def _step_num(path):
        m = re.search(r"step_(\d+)\.zip$", path)
        return int(m.group(1)) if m else 0
    latest = max(files, key=_step_num)
    step = _step_num(latest)
    model_path = latest.replace(".zip", "")
    buf_path = os.path.join(ckpt_dir, f"replay_buffer_{step}")
    return model_path, buf_path, step


# ── Environment Preview ────────────────────────────────────────────────────────
def view():
    """Spawns the biped + stair in GUI mode. Press Ctrl+C to quit."""
    import pybullet_data

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8, physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)

    assest = os.path.join(os.path.dirname(__file__), "assest")
    p.loadURDF(os.path.join(assest, "biped_.urdf"), [0, 0, 0.81],
               useFixedBase=False, physicsClientId=cid)
    p.loadURDF(os.path.join(assest, "stair.urdf"),  [0, 2, 0],
               p.getQuaternionFromEuler([0, 0, -3.1416]),
               useFixedBase=True, physicsClientId=cid)

    print("[view] Biped + stair spawned. Press Ctrl+C to quit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        pass
    p.disconnect(cid)


# ── Training ──────────────────────────────────────────────────────────────────
def train(timesteps: int, render: bool = False, sac_config: dict = None,
          config_name: str = None, resume: bool = False,
          checkpoint_freq: int = 50_000):
    """
    Trains a SAC agent on the 1 m platform jump task and saves the model.

    Steps
    -----
    1. Create training and evaluation environments (wrapped in Monitor).
    2. Instantiate SAC with SAC_CONFIG (or load from checkpoint if --resume).
    3. Set up EvalCallback, CheckpointCallback, and RewardPlotCallback.
    4. Call model.learn() and handle KeyboardInterrupt for crash-saves.
    5. Save the final model and plot the reward curve.
    """
    if sac_config is None:
        sac_config = SAC_CONFIG

    # Use config-specific directories when a config name is provided
    tag = f"_{config_name}" if config_name else ""
    tb_log_dir   = f"logs/sac_goal{tag}/"
    eval_log_dir = f"logs/sac_eval{tag}/"
    best_model_dir = f"./models/sac_best{tag}/"
    monitor_file = f"./logs/sac_monitor{tag}"

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # Checkpoint directory for this config
    ckpt_dir = f"./models/checkpoints{tag}/"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create training and evaluation environments
    train_env = Monitor(BipedJumpEnv(render=render), filename=monitor_file)
    eval_env  = Monitor(BipedJumpEnv(render=False))

    # Instantiate or resume SAC model
    if resume:
        model_path, buf_path, step = _find_latest_checkpoint(ckpt_dir)
        if model_path is None:
            # Fallback: try the best-model or crash-save
            fallbacks = [
                os.path.join(best_model_dir, "best_model"),
                f"models/sac_biped_crashsave{tag}",
            ]
            for fb in fallbacks:
                if os.path.exists(fb + ".zip"):
                    model_path = fb
                    step = 0
                    break
        if model_path:
            print(f"\n  [resume] Loading checkpoint: {model_path} (step {step})")
            model = SAC.load(model_path, env=train_env, tensorboard_log=tb_log_dir)
            # Restore replay buffer if available
            if buf_path and os.path.exists(buf_path + ".pkl"):
                print(f"  [resume] Loading replay buffer: {buf_path}")
                model.load_replay_buffer(buf_path)
            else:
                print(f"  [resume] No replay buffer found — starting fresh buffer")
        else:
            print("\n  [resume] No checkpoint found — starting from scratch")
            model = SAC(**sac_config, env=train_env, tensorboard_log=tb_log_dir)
            resume = False   # so reset_num_timesteps=True below
    else:
        model = SAC(**sac_config, env=train_env, tensorboard_log=tb_log_dir)

    # Callbacks
    reward_cb = RewardPlotCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=ckpt_dir,
    )

    print(f"\n{'='*60}")
    print(f"  Training SAC — {config_name or 'default'} | {timesteps:,} timesteps")
    print(f"  Resume      : {resume}")
    print(f"  Checkpoints : every {checkpoint_freq:,} steps → {ckpt_dir}")
    print(f"  TensorBoard : tensorboard --logdir logs/")
    print(f"  Best model  : {best_model_dir}")
    print(f"{'='*60}\n")

    # Training loop
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[eval_cb, reward_cb, ckpt_cb],
            reset_num_timesteps=not resume,
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving crash checkpoint...")
        model.save(f"models/sac_biped_crashsave{tag}")

    # Save final model and reward plot
    model.save("models/sac_biped_goal")
    reward_curve_path = f"reward_curve_sac{tag}.png"
    reward_cb.plot_rewards(save_path=reward_curve_path)
    print(f"[train] Final model saved to models/sac_biped_goal.zip")
    print(f"[train] Reward curve saved to {reward_curve_path}")

    train_env.close()
    eval_env.close()


# ── Evaluation ────────────────────────────────────────────────────────────────
def test(model_path: str, episodes: int, render: bool):
    """
    Loads a trained SAC model and evaluates it for a given number of episodes.

    Metrics reported per episode
    ----------------------------
    - Steps taken
    - Total reward
    - Energy consumed  (sum of |torque × velocity| × dt)
    - Distance travelled (Euclidean, spawn → landing)

    Summary metrics printed at the end
    -----------------------------------
    - Average reward
    - Fall rate  (%)
    - Average distance (m)
    - Average energy (J)
    - Cost of Transport (CoT) = Energy / (mass × g × distance)
    """
    DT = 1.0 / 50.0   # simulation timestep (must match utils.py)

    # Create environment and load model
    env = BipedJumpEnv(render=render)
    model = SAC.load(model_path, env=env)

    # Get joint indices for energy calculation
    joint_indices = env.get_joint_indices()

    # Accumulators
    total_energy   = 0.0
    total_distance = 0.0
    total_reward   = 0.0
    fall_count     = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward  = 0.0
        ep_energy  = 0.0
        ep_steps   = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps  += 1

            # Compute energy: sum(|torque * velocity| * dt)
            for j in joint_indices:
                js = p.getJointState(env.robot_id, j,
                                     physicsClientId=env.physics_client)
                torque   = js[3]   # applied motor torque
                velocity = js[1]   # joint velocity
                ep_energy += abs(torque * velocity) * DT

        # Distance from spawn to final position
        init_pos = env.robot_initial_position()
        final_pos = env.robot_current_position()
        dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(init_pos, final_pos)))

        total_reward   += ep_reward
        total_energy   += ep_energy
        total_distance += dist

        # Detect falls: extreme tilt at end of episode
        _, orn = p.getBasePositionAndOrientation(
            env.robot_id, physicsClientId=env.physics_client)
        euler = p.getEulerFromQuaternion(orn)
        tilt = abs(euler[0]) + abs(euler[1])
        if tilt > 1.0 or final_pos[2] < 0.3:
            fall_count += 1

        print(f"  Episode {ep + 1}/{episodes}: "
              f"steps={ep_steps}  reward={ep_reward:.2f}  "
              f"energy={ep_energy:.2f} J  dist={dist:.3f} m")

    # Summary
    n = episodes
    avg_reward   = total_reward / n
    fall_rate    = 100.0 * fall_count / n
    avg_distance = total_distance / n
    avg_energy   = total_energy / n
    cot          = total_energy / (ROBOT_MASS_KG * 9.81 * total_distance + 1e-8)

    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Average Reward    : {avg_reward:.2f}")
    print(f"  Fall Rate         : {fall_rate:.1f} %")
    print(f"  Avg Distance (m)  : {avg_distance:.4f}")
    print(f"  Avg Energy (J)    : {avg_energy:.2f}")
    print(f"  Cost of Transport : {cot:.4f}")
    print("=" * 55)

    env.close()


# ── CLI entry-point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3 — Biped 1 m Platform Jump (SAC)"
    )
    parser.add_argument("--mode",       choices=["view", "train", "test"], required=True,
                        help="view: preview env  |  train: train SAC  |  test: evaluate")
    parser.add_argument("--timesteps",  type=int, default=None,
                        help="Override TOTAL_TIMESTEPS from utils.py")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved model (.zip) for --mode test")
    parser.add_argument("--episodes",   type=int, default=EVAL_EPISODES,
                        help=f"Evaluation episodes (default: {EVAL_EPISODES})")
    parser.add_argument("--render",     action="store_true",
                        help="Enable PyBullet GUI")
    parser.add_argument("--config",     type=str, default=None,
                        help="Config key from configs.json (e.g. config_1, config_2, config_3)")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from the latest checkpoint")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000,
                        help="Save a checkpoint every N timesteps (default: 50000)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "view":
        view()
    elif args.mode == "train":
        ts = args.timesteps if args.timesteps else TOTAL_TIMESTEPS
        sac_config = None
        if args.config:
            with open(os.path.join(os.path.dirname(__file__), "configs.json")) as f:
                all_configs = json.load(f)
            if args.config not in all_configs:
                print(f"Error: config '{args.config}' not found. Available: {list(all_configs.keys())}")
                return
            cfg = all_configs[args.config]
            print(f"\n[config] Using '{cfg.get('name', args.config)}': {cfg}\n")
            sac_config = dict(
                policy        = "MlpPolicy",
                learning_rate = cfg["learning_rate"],
                buffer_size   = cfg["buffer_size"],
                batch_size    = cfg["batch_size"],
                tau           = cfg["tau"],
                gamma         = cfg["gamma"],
                ent_coef      = cfg["ent_coef"],
                verbose       = 1,
            )
            if not args.timesteps and "timesteps" in cfg:
                ts = cfg["timesteps"]
        train(ts, args.render, sac_config, config_name=args.config,
              resume=args.resume, checkpoint_freq=args.checkpoint_freq)
    elif args.mode == "test":
        if args.model_path is None:
            args.model_path = "models/sac_best/best_model"
        test(args.model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()
