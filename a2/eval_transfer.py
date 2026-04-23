import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from user_code import evaluate_policy, NUM_EPISODES, run_q_learning

def smooth(data, window=20):
    if len(data) < window: return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def main():
    print("====================================================")
    print("  DOMAIN TRANSFER TEST: Train at 1.0, Eval at 0.5")
    print("====================================================")
    
    # 1. Train at z=1.0
    print("\n[Stage 1] Training drone on TARGET_POS = [0, 0, 1.0]")
    env_train = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False, target_pos=[0, 0, 1.0])
    q_table, ep_rewards = run_q_learning(env_train, num_episodes=500)
    
    mean_train, std_train = evaluate_policy(env_train, q_table, num_episodes=10)
    print(f"  → Validation on [0, 0, 1.0]: {mean_train:.2f} ± {std_train:.2f}\n")
    env_train.close()
    
    # 2. Evaluate exactly same policy at z=0.5
    print("[Stage 2] Evaluating EXACT SAME previously trained policy on TARGET_POS = [0, 0, 0.5]")
    env_eval = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=True, target_pos=[0, 0, 0.5])
    mean_eval, std_eval = evaluate_policy(env_eval, q_table, num_episodes=10)
    print(f"  → Transfer Eval on [0, 0, 0.5]: {mean_eval:.2f} ± {std_eval:.2f}\n")
    env_eval.close()
    
    # Plotting
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8,5), dpi=150)
    plt.plot(ep_rewards, alpha=0.2, color='blue', label='Raw Rewards')
    plt.plot(np.arange(20, 20+len(smooth(ep_rewards))), smooth(ep_rewards), color='blue', linewidth=2, label='Smoothed (window=20)')
    plt.title("Q-Learning Training (Target=1.0)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join("results", "domain_transfer_train_curve.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")
    
    print("====================================================")

if __name__ == "__main__":
    main()
