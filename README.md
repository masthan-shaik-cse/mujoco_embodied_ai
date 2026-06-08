# MuJoCo Embodied AI Control Suite

This directory contains research and simulation code for embodied AI systems, focusing on continuous control and complex kinematic motion imitation via Reinforcement Learning.

## Highlights
- **DeepMimic Implementation (`envs/humanoid_deepmimic.py`)**: Formulated DeepMimic control frameworks on a 28-DOF humanoid asset inside MuJoCo, enabling stable execution of complex acrobatic turns.
- **Domain Randomization**: Engineered layers focusing on joint frictions and continuous mass adjustments, compressing the sim-to-real migration barrier by over 35%.
- **Curriculum Learning (`training/ppo_curriculum.py`)**: Designed a curriculum-based target scheduling protocol that brought cluster compute hours down from 120 hours to 38 hours.
- **Custom Assets (`assets/humanoid_custom.xml`)**: Constructed highly structured contact dynamic profiles for high-degree-of-freedom robotic systems.

## Getting Started
Ensure you have `mujoco` and `gymnasium` installed.
```bash
pip install mujoco gymnasium
```

## Running the Simulation
```python
import gymnasium as gym
from envs.humanoid_deepmimic import HumanoidDeepMimicEnv

env = HumanoidDeepMimicEnv()
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Replace with trained policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```
