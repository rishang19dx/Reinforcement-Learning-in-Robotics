import time
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from user_code import run_q_learning, discretize_state, extract_position, format_action, MAX_STEPS

def main():
    print("====================================================")
    print("  SLOW-MOTION VISUAL DOMAIN TRANSFER EVALUATION")
    print("====================================================")
    
    # 1. Train at z=1.0 headless
    print("[1] Training drone rapidly on TARGET_POS = [0, 0, 1.0]")
    env_train = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False, target_pos=[0, 0, 1.0])
    q_table, ep_rewards = run_q_learning(env_train, num_episodes=500)
    env_train.close()
    
    # 2. Evaluate with GUI and RECORDING precisely at z=0.5
    print("\n[2] Firing up PyBullet GUI to record evaluation carefully...")
    env_eval = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=True, target_pos=[0, 0, 0.5])
    
    # Extract client and build specific visual tracking markers!
    client = env_eval.CLIENT
    
    # Z = 0 (Red line & text)
    p.addUserDebugLine([-1, 0, 0], [1, 0, 0], [1, 0, 0], lineWidth=4, physicsClientId=client)
    p.addUserDebugText("z = 0.0 (Floor)", [-1.1, 0, 0], [1, 0, 0], physicsClientId=client, textSize=1.5)
    
    # Z = 0.5 (Green line & text)
    p.addUserDebugLine([-1, 0, 0.5], [1, 0, 0.5], [0, 1, 0], lineWidth=5, physicsClientId=client)
    p.addUserDebugText("z = 0.5 (NEW TARGET)", [-1.1, 0, 0.5], [0, 1, 0], physicsClientId=client, textSize=1.5)
    
    # Z = 1.0 (Blue line & text)
    p.addUserDebugLine([-1, 0, 1.0], [1, 0, 1.0], [0, 0.5, 1], lineWidth=4, physicsClientId=client)
    p.addUserDebugText("z = 1.0 (OLD TARGET)", [-1.1, 0, 1.0], [0, 0.5, 1], physicsClientId=client, textSize=1.5)
    
    # Slow down loop evaluation to create clear video
    print("\n   [->] Running frame-by-frame evaluation...")
    state, _ = env_eval.reset()
    state = discretize_state(extract_position(state), target_pos=env_eval.TARGET_POS)
    
    total_reward = 0
    for step in range(MAX_STEPS):
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env_eval.step(format_action(action))
        
        # This converts absolute position -> relative error
        state = discretize_state(extract_position(next_state), target_pos=env_eval.TARGET_POS)
        total_reward += reward
        
        if terminated or truncated:
            break
            
    print(f"Eval Total Reward: {total_reward:.2f}")
    
    env_eval.close()
    print("Video recording completed!")

if __name__ == "__main__":
    main()
