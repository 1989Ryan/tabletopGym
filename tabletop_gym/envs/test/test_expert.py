import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from PIL import Image
import numpy as np
from expert.motion import pick_and_place_action

from pybullet_env import Tabletop_Sim
from expert.TAMP_Solver import TAMP_Solver as ts
from utils import HEIGHT, SCENE_DIR
import random, os

task_solver = ts('focused')

file = random.choice(os.listdir(SCENE_DIR + 'train'))
state, table, step = map(int, file.split('_'))
# state = 427
# table = 304
# step = 12

sim = Tabletop_Sim(
    from_bullet=True, 
    state_id=state, 
    table_id = table, 
    step_id=step, 
    mode='train',
    width=300,
    height=300,
    use_gui=True,
)
planner = pick_and_place_action()

goal = task_solver.read_goal(
    state_id=state, 
    table_id = table, 
    step_id=step, 
    mode='train',
)
# print(goal)

init_state = sim.get_state()
solution = task_solver.get_solution(
    state_id=state, 
    table_id = table, 
    step_id=step, 
    mode='train',
    initial_states=init_state,
    temp_pose=[((0.0, 0.75, HEIGHT), (0.0,0.0,0.0,1.0))],
)
input()
plan = task_solver.resolve_plan(get_obs=sim.get_observation, step=sim.apply_action)
print(plan)
input()