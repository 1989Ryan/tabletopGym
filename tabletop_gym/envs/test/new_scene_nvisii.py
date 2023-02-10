import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from PIL import Image
import numpy as np
# from expert.motion import pick_and_place_action
import os
from pb_env_nvisii import Tabletop_Sim
import nvisii

sim = Tabletop_Sim(
    from_bullet=False, 
    state_id = 14, 
    table_id = 407, 
    step_id= 10, 
    mode='test',
    width=640,
    height=640,
    use_gui=False,
    obj_number=11,
    record_video=False,
)
sim.get_observation_nvisii("./exps/1/")
import random
ids = random.choice(sim.ids_pybullet_and_nvisii_names)
while ids['pybullet_id'] in [3, 4]:
    ids = random.choice(sim.ids_pybullet_and_nvisii_names)

sim.reset_obj_pose(ids['pybullet_id'], ids['nvisii_id'], (0, 0, 1.2,), 40)

# sim._dummy_run()
sim.get_observation_nvisii("./exps/2/")
# print('reseting')
# sim.reset_with_param(14, 407, 7, 'test', obj_number=11)
# sim.get_observation_nvisii("output1.png")
# print(sim.object_property)
# print('reseted')

# print('reseting')
# sim.reset_with_param(14, 407, 9, 'test', obj_number=11)
# sim.get_observation_nvisii("output2.png")
# print(sim.object_property)
# print('reseted')
nvisii.deinitialize()
                   
