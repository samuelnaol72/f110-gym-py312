# MIT License
# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng
# ... (Permission notice omitted for brevity)

# --- UPGRADED IMPORTS ---
import gymnasium as gym
# --- END UPGRADED IMPORTS ---

# --- EXPORT CORE MODULES ---
from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.dynamic_models import *
from f110_gym.envs.laser_models import *
from f110_gym.envs.base_classes import *
from f110_gym.envs.collision_models import *

# Register the F1TENTH environment using the Gymnasium API
gym.register(
    id='f110_gym:f110-v0',
    entry_point='f110_gym.envs.f110_env:F110Env',
    # The environment itself is now compliant, so the checker is not strictly necessary
    disable_env_checker=False 
)