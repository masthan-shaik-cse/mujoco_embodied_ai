import torch
import torch.nn as nn

class InverseRLPreferenceModel(nn.Module):
    """
    Inverse Reinforcement Learning (IRL) for Embodied Preferences.
    Robotic alignment requires human-preferred kinematics (e.g., a smooth walk 
    vs an erratic, energy-inefficient sprint). This module acts as a proxy reward 
    model trained on Bradley-Terry human preference comparisons.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a scalar 'preference score'
        )
        print("Initialized Embodied Preference Model (Inverse RL Aligner).")

    def predict_preference_score(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculates the human-aligned preference score for a given state-action pair.
        During training, this score is blended with the standard environment reward.
        """
        state_action = torch.cat([state, action], dim=-1)
        return self.net(state_action)

    def bradley_terry_loss(self, score_preferred: torch.Tensor, score_rejected: torch.Tensor) -> torch.Tensor:
        """
        Standard RLHF preference loss used to train the proxy model.
        """
        # Loss = -log(sigmoid(score_preferred - score_rejected))
        return -torch.nn.functional.logsigmoid(score_preferred - score_rejected).mean()
