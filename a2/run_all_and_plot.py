"""
Run RL algorithms (MC and Q-Learning),
collect quantitative results, and plot learning curves.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import sys
import json

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Import user_code algorithms
from user_code import (
    run_monte_carlo, run_q_learning, evaluate_policy,
    NUM_EPISODES, EPSILON, GAMMA, ALPHA
)

# ========================================
# CONFIGURATION
# ========================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SMOOTHING_WINDOW = 20  # for rolling average

def smooth(data, window=SMOOTHING_WINDOW):
    """Compute rolling average for smoother curves."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def create_env():
    return HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)

def run_all_experiments():
    """Train MC and Q-Learning algorithms and collect results."""

    results = {}

    # ---- 1. Monte Carlo ----
    print("=" * 60)
    print("  1/2  TRAINING: Monte Carlo Control")
    print("=" * 60)
    env = create_env()
    q_mc, rewards_mc = run_monte_carlo(env, num_episodes=NUM_EPISODES)
    mean_mc, std_mc = evaluate_policy(env, q_mc)
    env.close()
    results["Monte Carlo"] = {
        "rewards": rewards_mc,
        "eval_mean": float(mean_mc),
        "eval_std": float(std_mc),
        "final_50_avg": float(np.mean(rewards_mc[-50:])),
    }
    print(f"  → Eval: {mean_mc:.2f} ± {std_mc:.2f}\n")

    # ---- 2. Q-Learning ----
    print("=" * 60)
    print("  2/2  TRAINING: Q-Learning")
    print("=" * 60)
    env = create_env()
    q_ql, rewards_ql = run_q_learning(env, num_episodes=NUM_EPISODES)
    mean_ql, std_ql = evaluate_policy(env, q_ql)
    env.close()
    results["Q-Learning"] = {
        "rewards": rewards_ql,
        "eval_mean": float(mean_ql),
        "eval_std": float(std_ql),
        "final_50_avg": float(np.mean(rewards_ql[-50:])),
    }
    print(f"  → Eval: {mean_ql:.2f} ± {std_ql:.2f}\n")

    # Add convergence calculation
    for name in results:
        r = results[name]["rewards"]
        if len(r) >= 50:
            results[name]["convergence_episode"] = int(np.argmax([np.mean(r[i:i+50]) for i in range(len(r)-50)]))
        else:
            results[name]["convergence_episode"] = 500

    return results

def print_summary_table(results):
    """Print a nicely formatted summary table."""
    print("\n" + "=" * 90)
    print("  QUANTITATIVE RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'Algorithm':<22} {'Eval Mean':>12} {'Eval Std':>12} {'Last-50 Avg':>14} {'Conv. Ep':>12}")
    print("  " + "-" * 75)
    for name, r in results.items():
        print(f"  {name:<22} {r['eval_mean']:>12.2f} {r['eval_std']:>12.2f} {r['final_50_avg']:>14.2f} {r['convergence_episode']:>12}")
    print("=" * 90)

def plot_learning_curves(results):
    """Generate the requested individual and combined plots."""

    colors = {
        "Monte Carlo": "#E63946",        # Red
        "Q-Learning": "#457B9D",         # Steel blue
    }

    plot_paths = []

    # Helper function for plotting
    def create_plot(names, filename, title):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for name in names:
            r = results[name]
            # Raw data (transparent)
            episodes = np.arange(1, len(r["rewards"]) + 1)
            ax.plot(episodes, r["rewards"], alpha=0.15, color=colors[name], linewidth=0.5)
            # Smoothed data
            smoothed = smooth(r["rewards"])
            ep_smooth = np.arange(SMOOTHING_WINDOW, SMOOTHING_WINDOW + len(smoothed))
            ax.plot(ep_smooth, smoothed, color=colors[name], linewidth=2.5, label=f"{name} (smoothed)")

        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=13)
        ax.set_ylabel("Total Reward", fontsize=13)
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {path}")
        plot_paths.append(path)

    # 1. MC Separately
    create_plot(["Monte Carlo"], "learning_curve_mc_only.png", "Monte Carlo Control Learning Curve")
    
    # 2. TD Separately
    create_plot(["Q-Learning"], "learning_curve_td_only.png", "Q-Learning (TD) Learning Curve")

    # 3. Together
    create_plot(["Monte Carlo", "Q-Learning"], "learning_curve_mc_vs_td.png", "Monte Carlo vs Q-Learning")

    return plot_paths

def save_results_json(results):
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "eval_mean": r["eval_mean"],
            "eval_std": r["eval_std"],
            "final_50_avg": r["final_50_avg"],
            "convergence_episode": r["convergence_episode"],
        }
    path = os.path.join(RESULTS_DIR, "quantitative_results_mc_td.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {path}")
    return path

if __name__ == "__main__":
    print("\n" + "▓" * 60)
    print("  DRONE HOVER RL — MC AND TD RUNNER")
    print("▓" * 60 + "\n")

    results = run_all_experiments()
    print_summary_table(results)

    print("\n📊 Generating learning curve plots...")
    plot_paths = plot_learning_curves(results)

    print("\n💾 Saving quantitative results...")
    json_path = save_results_json(results)

    print("\n✅ All done! Check the 'results/' directory for plots and data.")
