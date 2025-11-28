# MIT License
# ... (Copyright and Permission notice omitted for brevity)

'''
Renderer for the F1TENTH Gym environment
'''

# opengl stuff
import pyglet
from pyglet import gl
from pyglet.gl import * # Keeping the original GL import style for functionality

# other
import numpy as np
from PIL import Image
import yaml
import os
import time

# helpers (Using ABSOLUTE PATH for Python 3.12)
# FIX: Use absolute import path for internal F1TENTH modules
from f110_gym.envs.collision_models import get_vertices 
# FIX: The rendering module needs the base classes for constants
from f110_gym.envs.base_classes import Integrator 

# constants
MAP_IMG_SIZE = 1000

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

# rendering constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

# vehicle geometry vertices
CAR_VERTICES = get_vertices(CAR_LENGTH, CAR_WIDTH)

class EnvRenderer:
    def __init__(self, width, height, map_name=None, map_ext=None):
        self.width = width
        self.height = height

        self.window = pyglet.window.Window(width=width, height=height, caption='F1TENTH', resizable=True)
        self.window.push_handlers(self)
        
        self.map_img = None
        self.map_name = map_name
        self.map_ext = map_ext
        self.map_path = None
        
        if self.map_name is None:
            # Default map path (assuming f110_env.py structure)
            self.map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps', 'vegas.yaml')
        else:
            self.map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps', self.map_name + '.yaml')

        self.map_img_path = None
        self.map_metadata = None
        self.map_ratio = 1.0

        self.map_batch = pyglet.graphics.Batch()
        self.map_sprite = None

        # race info
        self.score_label = pyglet.text.Label(
            'Time: 0.00\nLaps: 0\nStatus: Running',
            font_name='Times New Roman', font_size=18,
            x=20, y=self.height - 70, anchor_x='left', anchor_y='bottom',
            batch=pyglet.graphics.Batch()
        )
        self.fps_display = pyglet.window.FPSDisplay(window=self.window)
        
        # view
        self.left = 0
        self.right = self.width
        self.bottom = 0
        self.top = self.height
        self.zoom_level = 1.0
        self.zoomed_width = self.width
        self.zoomed_height = self.height
        
        # agent obs
        self.obs = None
        self.ego_idx = 0
        self.car_batches = []
        self.car_shapes = []
        self.laser_shapes = []
        
        if self.map_name is not None and self.map_ext is not None:
            self.update_map(self.map_name, self.map_ext)

    def on_resize(self, width, height):
        self.width = width
        self.height = height
        self.zoomed_width = self.width / self.zoom_level
        self.zoomed_height = self.height / self.zoom_level
        
        glViewport(0, 0, width, height)
        self.on_draw()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        f = ZOOM_IN_FACTOR if scroll_y < 0 else ZOOM_OUT_FACTOR
        
        self.zoom_level *= f
        self.zoomed_width = self.width / self.zoom_level
        self.zoomed_height = self.height / self.zoom_level
        
        # recalculate view center for zoom
        mouse_x = self.left + x / self.width * self.zoomed_width
        mouse_y = self.bottom + y / self.height * self.zoomed_height
        
        self.left = mouse_x - self.zoomed_width / 2
        self.right = mouse_x + self.zoomed_width / 2
        self.bottom = mouse_y - self.zoomed_height / 2
        self.top = mouse_y + self.zoomed_height / 2

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.left -= dx / self.width * self.zoomed_width
        self.right -= dx / self.width * self.zoomed_width
        self.bottom -= dy / self.height * self.zoomed_height
        self.top -= dy / self.height * self.zoomed_height

    def on_close(self):
        self.window.close()
        
    def update_map(self, map_name, map_ext):
        """Loads map and updates view based on map dimensions."""
        
        if map_name in ['berlin', 'vegas', 'skirk', 'levine']:
            map_path_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps', map_name)
        else:
            # Assumes map_name is the full path without extension
            map_path_base = map_name 
            
        self.map_img_path = map_path_base + map_ext
        map_yaml_path = map_path_base + '.yaml'

        with open(map_yaml_path, 'r') as yaml_file:
            self.map_metadata = yaml.safe_load(yaml_file)
        
        img = Image.open(self.map_img_path).transpose(Image.FLIP_TOP_BOTTOM)
        self.map_img = pyglet.image.ImageData(
            img.width, img.height, 
            'L' if img.mode == 'L' else 'RGB', 
            img.tobytes(), 
            pitch=img.width if img.mode == 'L' else -img.width * 3
        )
        
        # Calculate map ratio and adjust view to center the map
        self.map_ratio = self.map_metadata['resolution'] * img.width / self.width
        
        # Center view on map (0,0)
        map_center_x = self.map_metadata['origin'][0] + self.map_metadata['resolution'] * img.width / 2
        map_center_y = self.map_metadata['origin'][1] + self.map_metadata['resolution'] * img.height / 2
        
        self.zoomed_width = self.width / self.zoom_level
        self.zoomed_height = self.height / self.zoom_level

        self.left = map_center_x - self.zoomed_width / 2
        self.right = map_center_x + self.zoomed_width / 2
        self.bottom = map_center_y - self.zoomed_height / 2
        self.top = map_center_y + self.zoomed_height / 2

        self.map_sprite = pyglet.sprite.Sprite(
            self.map_img, x=self.map_metadata['origin'][0], y=self.map_metadata['origin'][1], 
            batch=self.map_batch
        )
        self.map_sprite.scale_x = self.map_metadata['resolution'] * img.width / img.width
        self.map_sprite.scale_y = self.map_metadata['resolution'] * img.height / img.height

    def update_obs(self, obs):
        """Updates agent observations."""
        self.obs = obs
        if obs is not None:
            self.ego_idx = obs['ego_idx']

    def draw_car(self, x, y, theta, steer_angle, color):
        """Draws a single car as a polygon."""
        
        # Calculate rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Transform vertices
        vertices = R @ CAR_VERTICES + np.array([[x], [y]])
        
        # Create vertex list for pyglet
        v_list = ('v2f', vertices.T.flatten())
        
        # Draw the car body
        car_body = pyglet.graphics.vertex_list(
            4, 'v2f', vertices.T.flatten(), 
            ('c3f', color * 4)
        )
        car_body.draw(GL_QUADS)
        
        # Draw steering wheels (simplified)
        front_axle_x = x + CAR_LENGTH * np.cos(theta) / 2
        front_axle_y = y + CAR_LENGTH * np.sin(theta) / 2
        
        for side in [-1, 1]:
            wheel_center_x = front_axle_x + side * (CAR_WIDTH/2) * np.cos(theta + np.pi/2)
            wheel_center_y = front_axle_y + side * (CAR_WIDTH/2) * np.sin(theta + np.pi/2)
            
            wheel_angle = theta + steer_angle
            wheel_length = 0.1
            
            # Wheel vertices (line segment)
            wheel_start = np.array([wheel_center_x - wheel_length/2 * np.cos(wheel_angle),
                                    wheel_center_y - wheel_length/2 * np.sin(wheel_angle)])
            wheel_end = np.array([wheel_center_x + wheel_length/2 * np.cos(wheel_angle),
                                  wheel_center_y + wheel_length/2 * np.sin(wheel_angle)])
            
            pyglet.graphics.draw(2, GL_LINES, ('v2f', (wheel_start[0], wheel_start[1], wheel_end[0], wheel_end[1])), ('c3f', (0.5, 0.5, 0.5) * 2))

    def draw_laser(self, x, y, theta, scans, color):
        """Draws laser scan points."""
        
        if scans is None:
            return

        points = []
        scan_angles = np.linspace(-np.pi/4, np.pi/4, len(scans)) # Placeholder angles, real angles depend on F110Env params
        
        for i, dist in enumerate(scans):
            if dist < 30.0: # Max range filter
                angle = theta + scan_angles[i]
                points.append(x + dist * np.cos(angle))
                points.append(y + dist * np.sin(angle))
        
        if points:
            pyglet.graphics.draw(len(points) // 2, GL_POINTS, ('v2f', points), ('c3f', color * (len(points) // 2)))


    def on_draw(self):
        """Main rendering loop."""
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(self.left, self.right, self.bottom, self.top, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Draw map
        self.map_batch.draw()
        
        # Draw agents and lasers
        if self.obs is not None:
            num_agents = len(self.obs['poses_x'])
            
            for i in range(num_agents):
                x = self.obs['poses_x'][i]
                y = self.obs['poses_y'][i]
                theta = self.obs['poses_theta'][i]
                
                # Colors: Ego is green, Opponents are red
                color = (0.1, 0.7, 0.1) if i == self.ego_idx else (0.7, 0.1, 0.1)
                
                # Draw Laser Scans
                # Note: Scans data structure is assumed to be a list of lists/arrays
                if 'scans' in self.obs and len(self.obs['scans']) > i and self.obs['scans'][i] is not None:
                    self.draw_laser(x, y, theta, self.obs['scans'][i], color)

                # Draw Car Body
                self.draw_car(x, y, theta, 0.0, color) # Steering angle is 0 here, should come from self.obs if available

            # Callbacks for extra drawing (e.g., waypoints, paths)
            for callback in self.render_callbacks:
                callback(self)


        # Draw text overlays (scores, FPS) in screen coordinates
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Update text info
        if self.obs is not None:
            time_str = f"{self.obs.get('lap_times', [0.0])[self.ego_idx]:.2f}"
            lap_count = int(self.obs.get('lap_counts', [0])[self.ego_idx])
            
            # Status check (assuming F1TENTH simulation logic)
            # This status check is simplified and may need tuning based on final F1TENTH collision data
            status = 'Running'
            if self.obs.get('collisions', [0])[self.ego_idx] == 1:
                status = 'CRASHED'
            elif lap_count >= 4: # Assuming 4 is the finish count
                status = 'Finished'

            self.score_label.text = (
                f'Time: {time_str}\n'
                f'Laps: {lap_count}\n'
                f'Status: {status}'
            )
        
        self.score_label.draw()
        self.fps_display.draw()

# Helper function to expose outside of class
def add_render_callback(callback_func):
    """
    Add extra drawing function to call during rendering.
    """
    if EnvRenderer.renderer is not None:
        EnvRenderer.render_callbacks.append(callback_func)
    else:
        # If renderer isn't initialized yet, store the callback
        print("Warning: Renderer not initialized. Callback will be registered on init.")