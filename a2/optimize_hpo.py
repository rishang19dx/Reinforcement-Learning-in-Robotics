import optuna
import numpy as np
import gymnasium as gym
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Constants from original script
STATE_DIM = 3
MAX_STEPS = 240

def discretize_state(state, num_bins):
    state = np.asarray(state)
    if state.ndim == 2:
        state = state[0, 0:3]
    else:
        state = state[0:3]

    bounds = np.array([[-1, 1], [-1, 1], [0, 2]])
    discrete = []
    for val, (low, high) in zip(state, bounds):
        val = np.clip(val, low, high)
        normalized = (val - low) / (high - low)
        bin_idx = int(normalized * num_bins)
        bin_idx = min(bin_idx, num_bins - 1)
        discrete.append(bin_idx)
    return tuple(discrete)

def get_action_space_size():
    return 3

def action_index_to_value(action_idx):
    return float(action_idx - 1)

def format_action(action):
    return np.array([[action_index_to_value(action)]], dtype=np.float32)

def extract_position(obs):
    obs_arr = np.asarray(obs)
    if obs_arr.ndim == 2:
        return obs_arr[0, 0:3]
    return obs_arr[0:3]

def evaluate_policy(env, q_table, num_bins, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(extract_position(state), num_bins)
        total_reward = 0
        for _ in range(MAX_STEPS):
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_state), num_bins)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards)

def objective(trial):
    # Retrieve hyperparameters from trial
    try:
        env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False, record=False)
    except Exception as e:
        print("Error initializing env:", e)
        return -1000.0

    num_bins = trial.suggest_int('NUM_BINS', 8, 15)
    epsilon = trial.suggest_float('EPSILON', 0.05, 0.3)
    gamma = trial.suggest_float('GAMMA', 0.95, 0.999)
    alpha = trial.suggest_float('ALPHA', 0.05, 0.2)
    num_episodes = trial.suggest_int('NUM_EPISODES', 500, 1000)

    shape = (num_bins,) * STATE_DIM + (get_action_space_size(),)
    q_table = np.zeros(shape)
    
    # Pruning setup
    eval_freq = max(1, num_episodes // 5)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = discretize_state(extract_position(obs), num_bins)
        
        for _ in range(MAX_STEPS):
            if np.random.random() < epsilon:
                action = np.random.randint(get_action_space_size())
            else:
                action = np.argmax(q_table[state])
                
            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_obs), num_bins)
            
            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q
            q_table[state][action] += alpha * (td_target - q_table[state][action])
            
            state = next_state
            if terminated or truncated:
                break

        # Periodic intermediate evaluation for pruning
        if (episode + 1) % eval_freq == 0:
            intermediate_score = evaluate_policy(env, q_table, num_bins, num_episodes=5)
            trial.report(intermediate_score, episode)
            if trial.should_prune():
                env.close()
                raise optuna.exceptions.TrialPruned()

    score = evaluate_policy(env, q_table, num_bins, num_episodes=15)
    env.close()
    return score

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    # Suppress verbose pybullet outputs by redirecting stdout inside Optuna indirectly or just ignore
    print("Starting Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100))
    study.optimize(objective, n_trials=30)
    
    print("\\n\\n=========================================")
    print("BEST HYPERPARAMETERS FOUND:")
    print("=========================================")
    for key, val in study.best_params.items():
        print(f"{key}: {val}")
    print(f"Best Evaluation Score: {study.best_value}")
