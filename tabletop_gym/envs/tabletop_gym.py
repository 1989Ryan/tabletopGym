import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
import random
import os
# from gym import spaces
# import time
# import pybullet as p
# from . import kuka
import numpy as np
# import pybullet_data
# import pdb
# import distutils.dir_util
# import glob
from pkg_resources import parse_version
import gym
from pybullet_env import Tabletop_Sim
from instruction import Instruction
from utils import HEIGHT, WIDTH, SCENE_DIR
# from ur5_utils.pickplace import PickPlaceContinuous
# from gym.envs.classic_control import rendering

# WINDOW_W = 1000
# WINDOW_H = 800

class TabletopGymEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self,
                use_gui=False,
                mode='train',
               ):
        self.mode = mode
        self.use_gui = use_gui

        print("TabletopGymEnv __init__")
        color_tuple = [
            gym.spaces.Box(0, 255, (HEIGHT, WIDTH) + (3,), dtype=np.uint8)
            for _ in range(3)
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, (HEIGHT, WIDTH), dtype=np.float32)
            for _ in range(3)
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
            'instruction': Instruction(min_length=1, max_length=36),
            'ee_sensor': gym.spaces.Discrete(2),
        })
        self.position_bounds = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 1.10], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Dict({
            'move_cmd': # position and quaternion
                gym.spaces.Tuple(
                    (self.position_bounds,
                    gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))), 
            'suction_cmd': gym.spaces.Discrete(2),  # Binary 0-1.
            'acts_left': gym.spaces.Discrete(1000),
        })
        self._sim = Tabletop_Sim(from_bullet=True, use_gui=self.use_gui)
    
    def seed(self):
        '''
        add random seeds for picking random states in 
        training or testing
        '''
        file = random.choice(os.listdir(SCENE_DIR + self.mode))
        state, table, step = map(int, file.split('_'))
        
        return state, table, step
    
    def reset(self):
        state, table, step = self.seed()
        self._sim.reset_with_param(state_id=state, 
                                    table_id=table, 
                                    step_id=step, 
                                    mode=self.mode,
                                    from_bullet=True)
        return self._sim.get_observation()
    
    def _get_obs(self):
        rgb, depth, ins, ee_sensor = self._sim.get_observation()
        # obs = {}
        obs = {'color': rgb, 'depth': depth, 'instruction': ins, 'ee_sensor': ee_sensor}
        return obs
        
    def step(self, action):
        # TODO: implement the robot control and planning module for
        # pick and place tasks.
        if action is not None:
            timeout = self._sim.apply_action(action)
            if timeout:
                obs = self._get_obs()
                return obs, 0.0, True, self.info
        # print('I am working!')
        while not self._sim.is_static:
            self._sim.run()
    
        # Get task rewards.
        reward, info = self._sim.reward() if action is not None else (0, {})
        task_done = self._sim.done()
        if action is not None:
            done = task_done and action['acts_left'] == 0
        else:
            done = task_done

        # Add ground truth robot state into info.
        info.update(self.info)
        obs = self._get_obs()
        return obs, reward, done, info
    
    @property
    def info(self):
        return self._sim.get_info()

    def render(self, mode='human'):
        # self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
        rgb = self._sim.get_render_img()
        
        return rgb
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

