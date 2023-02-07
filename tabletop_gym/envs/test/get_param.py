from json import dump
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import pybullet as pb
from sklearn.linear_model import LinearRegression
from pybullet_env import Tabletop_Sim
from utils import dump_json, read_action_csv, read_json
from scipy.spatial.transform import Rotation
# sim = Tabletop_Sim()
unity_conf = read_action_csv('HIT_23_configs_102_0.csv')
pybullet_conf = read_json('conf.json')

Trans = {}
X = []
Y = []
for conf in unity_conf:
    conf_p = np.array(pybullet_conf[conf]['pose'])[:2]
    X.append(unity_conf[conf][:2])
    Y.append(conf_p)
X = np.array(X)
Y = np.array(Y)
reg = LinearRegression().fit(X, Y)
reg_R = np.matrix(reg.coef_)
reg_b = np.matrix(reg.intercept_).reshape(-1,1)


# print(reg.score(X, Y))
print(reg_R)
print(reg_b)
print(reg_R * np.matrix(X[0]).reshape(-1,1) + reg_b)
# print(Y[0])
Trans['A'] = reg_R.tolist()
Trans['b'] = reg_b.tolist()
dump_json(Trans, 'tabletop_gym/envs/Trans.json')


# X = []
# Y = []
Rotation_dict = dict()
for conf in unity_conf:
    # print(conf)
    X = pb.getQuaternionFromEuler([0., 0., unity_conf[conf][2]*np.pi/180.])
    Y = pb.getQuaternionFromEuler(pybullet_conf[conf]['orn'])
    # Diff = pb.getDifferenceQuaternion(X, Y)
    RX = np.matrix(pb.getMatrixFromQuaternion(X)).reshape(3,3)
    RY = np.matrix(pb.getMatrixFromQuaternion(Y)).reshape(3,3)
    R = RY * RX.I # * RX.I
    # print(np.matrix([0., 0., unity_conf[conf][2]*np.pi/180.]).reshape(-1, 1))
    # print(R)
    r = Rotation.from_matrix(R * RX)
    print(r.as_quat())
    print('ground truth: {}'.format(Y))
    Rotation_dict[conf] = R.tolist()
dump_json(Rotation_dict, 'tabletop_gym/envs/config/Rotation.json')