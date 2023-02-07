import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from utils import dump_json, read_action_csv, read_json, transform
import pybullet as p

unity_conf = read_action_csv('HIT_23_configs_102_0.csv')
pybullet_conf = read_json('conf.json')

for conf in unity_conf:
    pos, orn = transform(conf, unity_conf[conf])
    print(p.getEulerFromQuaternion(orn))
    print(pybullet_conf[conf]['orn'])