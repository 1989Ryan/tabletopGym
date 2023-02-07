import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from PIL import Image
import numpy as np
from expert.motion import pick_and_place_action
import os
from pybullet_env import Tabletop_Sim

sim = Tabletop_Sim(
    from_bullet=False, 
    state_id=143, 
    table_id = 104, 
    step_id=4, 
    mode='test',
    width=640,
    height=640,
    use_gui=False,
    obj_number=11,
    record_video=False,
)
img, _, _, _, _ = sim.get_observation_with_mask()
img = Image.fromarray(img[:, :, :3])
img.save('obs.png')
                   
