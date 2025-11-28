# MIT License
# ... (Copyright and Permission notice omitted for brevity but remain valid)

'''
Author: Hongrui Zheng (Original Code), Gemini (Gymnasium Upgrade)
'''

# --- UPGRADED IMPORTS ---
import os
from importlib.resources import files # <-- NEW, CRITICAL IMPORT
import gymnasium as gym
from gymnasium import spaces # Add or verify this import
import numpy as np
# --- END UPGRADED IMPORTS ---

# base classes
from f110_gym.envs.base_classes import Simulator, Integrator # <--- This is CORRECT!

# others
import numpy as np
import os
import time

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

# We use the original class name but implement the gymnasium API
class F110Env(gym.Env):
    """
    OpenAI Gymnasium environment for F1TENTH (Modern API Implementation)
    """
    metadata = {'render_modes': ['human', 'human_fast'], 'render_fps': 50}
    
    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        # --- Map Path Resource Setup ---
        # Define the base location for map files within the installed package
        BASE_MAP_PATH = files('f110_gym') / 'envs' / 'maps'
        
        # -------------------- MAP LOADING --------------------
        try:
            self.map_name = kwargs['map']
            
            # Determine map path based on known maps
            if self.map_name == 'berlin':
                resource_path = BASE_MAP_PATH / 'berlin.yaml'
            elif self.map_name == 'skirk':
                resource_path = BASE_MAP_PATH / 'skirk.yaml'
            elif self.map_name == 'levine':
                resource_path = BASE_MAP_PATH / 'levine.yaml'
            else:
                # Case: A custom map path was passed (assumed to be absolute/external)
                # We don't use importlib for external paths
                self.map_path = self.map_name + '.yaml'
                raise ValueError("Custom map name provided. Assuming map path is external and complete.")
                
            # Convert the resource path object to a string path for the Simulator
            if self.map_name in ['berlin', 'skirk', 'levine']:
                self.map_path = str(resource_path)
                
        except (KeyError, ValueError):
            # This handles: 
            # 1. No 'map' kwarg passed (KeyError)
            # 2. A custom map name was passed but failed (ValueError)
            
            # SET DEFAULT MAP: 'vegas' (using the stable resource path)
            resource_path = BASE_MAP_PATH / 'vegas.yaml'
            self.map_path = str(resource_path)
            self.map_name = 'vegas' # Set map_name for consistency

        # Map image extension
        self.map_ext = kwargs.get('map_ext', '.png')

        # Vehicle parameters
        self.params = kwargs.get(
            'params',
            {
                'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562,
                'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                'm': 3.74, 'I': 0.04712,
                's_min': -0.4189, 's_max': 0.4189,
                'sv_min': -3.2, 'sv_max': 3.2,
                'v_switch': 7.319, 'a_max': 9.51,
                'v_min': -5.0, 'v_max': 20.0,
                'width': 0.31, 'length': 0.58
            }
        )

        # Simulation parameters
        self.num_agents = kwargs.get('num_agents', 2)
        self.timestep = kwargs.get('timestep', 0.01)
        self.ego_idx = kwargs.get('ego_idx', 0)
        self.integrator = kwargs.get('integrator', Integrator.RK4)
        self.lidar_dist = kwargs.get('lidar_dist', 0.0)

        # -------------------- ACTION SPACE (Gymnasium-compatible) --------------------
        low_agent = np.array(
            [self.params['s_min'], self.params['v_min']],
            dtype=np.float32
        )
        high_agent = np.array(
            [self.params['s_max'], self.params['v_max']],
            dtype=np.float32
        )

        low = np.tile(low_agent, (self.num_agents, 1))
        high = np.tile(high_agent, (self.num_agents, 1))

        self.action_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        # -------------------- OBSERVATION SPACE --------------------
        self.observation_space = spaces.Dict({
            'ego_idx': spaces.Discrete(self.num_agents),
            'scans': spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_agents, 1080),
                dtype=np.float32
            ),
            'poses_x': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'poses_y': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'poses_theta': spaces.Box(
                low=-np.pi, high=np.pi,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'linear_vels_x': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'linear_vels_y': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'ang_vels_z': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'collisions': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'lap_times': spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
            'lap_counts': spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.num_agents,),
                dtype=np.float32
            ),
        })

        # -------------------- INTERNAL STATE VARIABLES --------------------
        self.start_thresh = 0.5 # 10 cm

        self.poses_x = np.zeros(self.num_agents)
        self.poses_y = np.zeros(self.num_agents)
        self.poses_theta = np.zeros(self.num_agents)
        self.collisions = np.zeros(self.num_agents)

        # Missing in your version → FIXED
        self.lap_times = np.zeros(self.num_agents)
        self.lap_counts = np.zeros(self.num_agents, dtype=int)
        self.time_steps = 0
        self.collision_idx = -1

        # Loop detection
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros(self.num_agents)
        self.start_xs = np.zeros(self.num_agents)
        self.start_ys = np.zeros(self.num_agents)
        self.start_thetas = np.zeros(self.num_agents)
        self.start_rot = np.eye(2)

        # -------------------- SIMULATOR INIT --------------------
        self.sim = Simulator(
            self.params,
            self.num_agents,
            12345, # seed moved to reset()
            time_step=self.timestep,
            integrator=self.integrator,
            lidar_dist=self.lidar_dist
        )
        # This call now uses the stable path set above
        self.sim.set_map(self.map_path, self.map_ext)

        # for rviz rendering
        self.render_obs = None


    def __del__(self):
        pass

    def _check_done(self):
        """ Check if the current rollout is done (Collision or Laps finished) """
        
        # --- Original _check_done logic ---
        left_t = 2
        right_t = 2
        
        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        # Check for episode termination (due to collision)
        terminated = (self.collisions[self.ego_idx] == 1)
        
        # Check for episode truncation (due to reaching lap goal)
        truncated = np.all(self.toggle_list >= 4)
        
        return terminated, truncated

    def _update_state(self, obs_dict):
        """ Update the env's states according to observations """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    # --- UPGRADED: Returns 5 values: obs, reward, terminated, truncated, info ---
    def step(self, action):
        """
        Step function for the gymnasium env (returns 5 values)
        """
        
        # call simulation step
        obs = self.sim.step(action)
        
        # Ensure all observations are in the correct dtype (float32)
        obs['scans'] = np.asarray(obs['scans'], dtype=np.float32)
        obs['poses_x'] = np.asarray(obs['poses_x'], dtype=np.float32)
        obs['poses_y'] = np.asarray(obs['poses_y'], dtype=np.float32)
        obs['poses_theta'] = np.asarray(obs['poses_theta'], dtype=np.float32)
        obs['linear_vels_x'] = np.asarray(obs['linear_vels_x'], dtype=np.float32)
        obs['linear_vels_y'] = np.asarray(obs['linear_vels_y'], dtype=np.float32)
        obs['ang_vels_z'] = np.asarray(obs['ang_vels_z'], dtype=np.float32)
        obs['collisions'] = np.asarray(obs['collisions'], dtype=np.float32)
        obs['lap_times'] = np.asarray(self.lap_times, dtype=np.float32)
        obs['lap_counts'] = np.asarray(self.lap_counts, dtype=np.float32)

        F110Env.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep
        
        # update data member
        self._update_state(obs)

        # check done (now split into terminated and truncated)
        terminated, truncated = self._check_done()

        info = {'checkpoint_done': truncated}

        # NOTE: Modern API returns 5 values: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

    # --- UPGRADED: Modern reset signature (seed=None, options=None) ---
    def reset(self, seed=None, options=None):
        """
        Reset the gymnasium environment.
        
        Args:
            seed (int, optional): The seed to reset the environment with.
            options (dict, optional): 
                'poses' (np.ndarray (num_agents, 3)): poses to reset agents to.
        
        Returns:
            obs (dict): initial observation of the current episode
            info (dict): auxillary information dictionary (always returned)
        """
        # 1. Handle seed via super().reset()
        super().reset(seed=seed)
        
        # 2. Handle Poses from Options
        if options is not None and 'poses' in options:
            poses = options['poses']
        else:
            # Default reset position if no pose is given 
            poses = np.array([
                [self.start_xs[i], self.start_ys[i], self.start_thetas[i]] 
                for i in range(self.num_agents)
            ])
            
        # 3. Handle Seed update for the Simulator (must be done before sim.reset)
        if seed is not None:
            # Re-initialize simulator with the new seed value from super().reset()
            # Note: self.sim expects the seed to be passed in __init__
            self.sim = Simulator(self.params, self.num_agents, seed, time_step=self.timestep, integrator=self.integrator, lidar_dist=self.lidar_dist)
            self.sim.set_map(self.map_path, self.map_ext)
            
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        
        # Call step() to get initial observation.
        # NOTE: step() returns 5 values (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.step(action)
        
        # Manually reset terminated/truncated flags after initial observation
        terminated = False
        truncated = False
        
        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }
        
        # NOTE: Modern reset returns 2 values: obs, info
        return obs, info

    def update_map(self, map_path, map_ext):
        """ Updates the map used by simulation """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """ Updates the parameters used by simulation for vehicles """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """ Add extra drawing function to call during rendering. """
        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """ Renders the environment with pyglet. """
        assert mode in ['human', 'human_fast']
        
        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_name, self.map_ext)
            
        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)
            
        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
