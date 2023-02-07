import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from PIL import Image
import numpy as np
from expert.motion import pick_and_place_action

from pybullet_env import Tabletop_Sim

# sim = Tabletop_Sim()
record_cfg = {
    'save_video_path': '/home/zirui/tabletop_gym/',
    'video_name': 'test.mp4',
    'fps': 30,
}
sim = Tabletop_Sim(
    from_bullet=False, 
    state_id=388, 
    table_id = 508, 
    step_id=7, 
    mode='train',
    width=640,
    height=640,
    use_gui=False,
    obj_number=11,
    record_video=True,
    record_cfg = record_cfg,
)
planner = pick_and_place_action()
# sim.save_scene()
img = sim.get_render_img()
# render_img = sim.get_render_img()

img = Image.fromarray(img[:, :, :3])
img.save('obs.png')
# render_img = Image.fromarray(render_img[:, :, :3])
# render_img.save('render.png')
input()
pos, orn = sim.get_obj_pose(5)
origin_orn = orn
pos1, pos2, pos3 = pos
pos3 = pos3 + 0.13
pos = (pos1, pos2, pos3)
pose = (pos, orn)
action = planner.pre_post_pick_place_from_top(pose)
sim.apply_action(action)
img, _, _, _ = sim.get_observation()
ee_sensor = sim.ee.detect_contact()
while not ee_sensor:
    # pos, orn = sim.get_obj_pose(18)
    pos1, pos2, pos3 = pos
    pos = (pos1, pos2, pos3-0.02)
    pose = (pos, orn)
    action=planner.pick_from_top(pose, ee_sensor)
    sim.apply_action(action)
    ee_sensor = sim.ee.detect_contact()
    img, _, _, _ = sim.get_observation()

pos1, pos2, pos3 = pos
pos = (pos1, pos2, pos3-0.02)
pose = (pos, orn)
action=planner.pick_from_top(pose, ee_sensor)
sim.apply_action(action)
img, _, _, _ = sim.get_observation()
img = Image.fromarray(img[:, :, :3])
img.save('obs1.png')
action=planner.pre_post_pick_place_from_top(pose)
sim.apply_action(action)
img, _, _, _ = sim.get_observation()

pos, orn = sim.get_obj_pose(4)
pos1, pos2, pos3 = pos
pos = (0.15, 0.12, 1.25)
pose = (pos, orn)
action=planner.pre_post_pick_place_from_top(pose)
sim.apply_action(action)
img, _, _, _ = sim.get_observation()
img = Image.fromarray(img[:, :, :3])
img.save('obs2.png')
ee_sensor = sim.ee.detect_contact()
while not ee_sensor:
    pos1, pos2, pos3 = pos
    pos = (pos1, pos2, pos3-0.02)
    pose = (pos, orn)
    action = planner.place_from_top(pose, ee_sensor, pose)
    sim.apply_action(action)
    ee_sensor = sim.ee.detect_contact()
    img, _, _, _ = sim.get_observation()

action=planner.place_from_top(pose, ee_sensor, pose)
sim.apply_action(action)
img, _, _, _ = sim.get_observation()
img = Image.fromarray(img[:, :, :3])
img.save('obs3.png')
action=planner.pre_post_pick_place_from_top(pose)
sim.apply_action(action)
img, _, _, _ = sim.get_observation()
img = Image.fromarray(img[:, :, :3])
img.save('obs4.png')
sim.close_video()
print("press any botton to continue: ")
input()