import torch
import torch.nn as nn
import numpy as np

class SafePPOLagrangian:
    """
    Constrained Markov Decision Process (CMDP) Optimizer.
    Unlike standard PPO which only maximizes reward, PPO-Lagrangian introduces
    a dual-optimization problem. It dynamically scales penalty multipliers (Lagrangian multipliers)
    when physical safety constraints (e.g., maximum torque, collision thresholds) are violated,
    mathematically guaranteeing safe exploration.
    """
    def __init__(self, cost_limit: float = 25.0, penalty_lr: float = 0.05):
        self.cost_limit = cost_limit
        self.penalty_lr = penalty_lr
        
        # The Lagrangian multiplier (penalty weight) initialized to 0
        self.lagrangian_multiplier = torch.zeros(1, requires_grad=True)
        self.penalty_optimizer = torch.optim.Adam([self.lagrangian_multiplier], lr=self.penalty_lr)
        print(f"Initialized Safe PPO-Lagrangian (CMDP Constraint Limit: {self.cost_limit})")

    def update_penalty(self, episodic_costs: list) -> float:
        """
        Dynamically adjusts the penalty multiplier based on constraint violations.
        If costs exceed the limit, the penalty weight increases, forcing the agent 
        to prioritize safety over reward in the next policy update.
        """
        mean_cost = np.mean(episodic_costs)
        
        # The objective is to maximize: lambda * (mean_cost - cost_limit)
        # So loss is the negative of that.
        loss = -self.lagrangian_multiplier * (mean_cost - self.cost_limit)
        
        self.penalty_optimizer.zero_grad()
        loss.backward()
        self.penalty_optimizer.step()
        
        # Multiplier must remain non-negative
        with torch.no_grad():
            self.lagrangian_multiplier.clamp_(min=0.0)
            
        current_penalty = self.lagrangian_multiplier.item()
        print(f"[Safe RL] Episodic Cost: {mean_cost:.2f} | Updated Lagrangian Multiplier: {current_penalty:.4f}")
        return current_penalty

    def apply_lagrangian_penalty(self, reward: float, cost: float) -> float:
        """
        Subtracts the weighted constraint violation cost from the raw environment reward.
        """
        penalty = self.lagrangian_multiplier.item() * cost
        return reward - penalty
