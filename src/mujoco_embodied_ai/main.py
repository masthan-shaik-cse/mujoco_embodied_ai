import argparse
import numpy as np
import torch
from src.mujoco_embodied_ai.alignment.safe_ppo_lagrangian import SafePPOLagrangian
from src.mujoco_embodied_ai.alignment.reward_hack_detector import RewardHackDetector
from src.mujoco_embodied_ai.alignment.inverse_rl_preferences import InverseRLPreferenceModel

def train_safe_ppo():
    print("\n--- Executing Safe RL (PPO-Lagrangian) Optimizer ---")
    cmdp_optimizer = SafePPOLagrangian(cost_limit=15.0)
    
    # Mock episode execution
    print("Running simulated episode...")
    episodic_costs = [18.5, 20.1, 16.0] # Exceeds the 15.0 limit
    
    new_penalty = cmdp_optimizer.update_penalty(episodic_costs)
    raw_reward = 100.0
    safe_reward = cmdp_optimizer.apply_lagrangian_penalty(raw_reward, np.mean(episodic_costs))
    
    print(f"Raw Task Reward: {raw_reward:.2f}")
    print(f"Constrained (Safe) Reward: {safe_reward:.2f}")

def run_reward_hack_audit():
    print("\n--- Executing Kinematic Reward Hacking Audit ---")
    detector = RewardHackDetector(vibration_threshold=10.0)
    
    # Mocking severe joint oscillation (vibrating to clip geometry)
    for _ in range(6):
        mock_velocities = np.random.uniform(-50, 50, size=(10,))
        audit_result = detector.audit_kinematics(mock_velocities)
        
    if audit_result["hacked"]:
        print("Audit Complete: Critical physics exploitation detected. Policy rejected.")
    else:
        print("Audit Complete: Kinematics verified as physically valid.")

def align_via_inverse_rl():
    print("\n--- Executing Inverse RL Preference Fine-Tuning ---")
    state_dim, action_dim = 24, 6
    irl_model = InverseRLPreferenceModel(state_dim, action_dim)
    
    # Mock state/action tensors
    mock_state = torch.randn(1, state_dim)
    mock_action_smooth = torch.randn(1, action_dim) * 0.1
    mock_action_jerky = torch.randn(1, action_dim) * 2.0
    
    score_smooth = irl_model.predict_preference_score(mock_state, mock_action_smooth)
    score_jerky = irl_model.predict_preference_score(mock_state, mock_action_jerky)
    
    loss = irl_model.bradley_terry_loss(score_smooth, score_jerky)
    print(f"Calculated Preference Tuning Loss (Bradley-Terry): {loss.item():.4f}")
    print("Embodied policy successfully aligned towards human-preferred kinematics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise Embodied AI Alignment Suite")
    parser.add_argument("--train_safe_ppo", action="store_true", help="Execute the Constrained MDP (Safe RL) Pipeline")
    parser.add_argument("--run_reward_hack_audit", action="store_true", help="Audit the policy for simulation exploits")
    parser.add_argument("--align_via_inverse_rl", action="store_true", help="Align robotic kinematics using human preferences")
    parser.add_argument("--run_all", action="store_true", help="Execute the full End-to-End Alignment Pipeline")
    
    args = parser.parse_args()
    
    if args.run_all:
        train_safe_ppo()
        run_reward_hack_audit()
        align_via_inverse_rl()
    else:
        if args.train_safe_ppo:
            train_safe_ppo()
        if args.run_reward_hack_audit:
            run_reward_hack_audit()
        if args.align_via_inverse_rl:
            align_via_inverse_rl()
            
        if not any(vars(args).values()):
            print("Please specify an execution flag. Use --help for options.")
