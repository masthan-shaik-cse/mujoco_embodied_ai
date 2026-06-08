import numpy as np

class RewardHackDetector:
    """
    Kinematic Anomaly Detection Node.
    RL agents often exploit simulation bugs (e.g., vibrating to gain momentum, 
    clipping through physical constraints) to maximize reward artificially.
    This module analyzes trajectory derivatives to detect and penalize 
    non-physical or "hacked" behaviors mathematically.
    """
    def __init__(self, vibration_threshold: float = 50.0):
        self.vibration_threshold = vibration_threshold
        self.history_buffer = []
        print("Initialized Kinematic Reward Hacking Detector.")

    def audit_kinematics(self, joint_velocities: np.ndarray) -> dict:
        """
        Analyzes the first and second derivatives of joint movements.
        High-frequency oscillations across multiple joints strongly indicate 
        physics engine exploitation.
        """
        self.history_buffer.append(joint_velocities)
        
        if len(self.history_buffer) < 5:
            return {"hacked": False, "penalty": 0.0}
            
        # Calculate jerk (derivative of acceleration/velocity)
        recent_vels = np.array(self.history_buffer[-5:])
        accelerations = np.diff(recent_vels, axis=0)
        jerks = np.diff(accelerations, axis=0)
        
        # High jerk magnitude indicates non-physical vibration (reward hacking)
        jerk_magnitude = np.linalg.norm(jerks)
        
        is_hacked = jerk_magnitude > self.vibration_threshold
        
        penalty = 0.0
        if is_hacked:
            penalty = 100.0 # Massive penalty for exploiting
            print(f"[ALIGNMENT WARNING] Reward Hacking Detected! (Jerk Magnitude: {jerk_magnitude:.2f}). Applying penalty.")
            
        return {"hacked": is_hacked, "penalty": penalty}
