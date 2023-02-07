import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from utils import read_action_csv, read_json, transform, inv_transform
import pybullet as p
unity_conf = read_action_csv('/home/zirui/tabletop_gym/tabletop_gym/envs/data/train/HIT_1_configs_102_0.csv')
pybullet_conf = read_json('/home/zirui/tabletop_gym/tabletop_gym/envs/scenes/train/1_102_0/info.json')

for conf in unity_conf:
    pos, orn = transform(conf, unity_conf[conf])
    # print(p.getEulerFromQuaternion(orn))
    # print(pybullet_conf[conf]['orn'])
    position, euler = inv_transform(conf, pos, orn)
    # print(position)
    print(euler)
    print(unity_conf[conf])