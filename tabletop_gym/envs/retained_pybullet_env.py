from json import dump
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)
import pybullet as pb
import pybullet_data
import numpy as np
# from tools.pybullet_tools.kuka_primitives import  BodyPose, BodyConf, Command, get_grasp_gen, \
#     get_stable_gen, get_ik_fn, get_free_motion_gen, \
#     get_holding_motion_gen, get_movable_collision_test, get_tool_link
# from tools.pybullet_tools.utils import DRAKE_IIWA_URDF, connect, load_model
from ur5_utils.gripper import Suction
from ur5_utils.pickplace import PickPlaceContinuous
from utils import *
import math, time
from pybullet_utils.bullet_client import BulletClient
# from pybullet_rendering.render.panda3d import P3dRenderer
from pybullet_rendering import RenderingPlugin
# from pybullet_rendering.render.panda3d import Mesh
# from pybullet_rendering.render.utils import primitive_mesh
from PIL import Image
from pybullet_rendering.render.pyrender import PyrRenderer
from timeit import default_timer as timer

class Tabletop_Sim:
    '''
    Tabletop gym pybullet physical environment.
    Load the physical environments according to the 
    config file for different scenarios.
    '''
    def __init__(self,
                from_bullet=True,
                table_id=None,
                state_id=None,
                step_id=None,
                mode='train',
                time_step=1./240.,
                renders=True,
                width=640,
                height=640,):
        '''
        config::scene configurations file path
        '''
        self._from_bullet = from_bullet
        self._state_id = state_id
        self._step_id = step_id
        self._table_id = table_id
        self.mode = mode
        self._renders = renders
        self._time_step = time_step
        self._obj_args = read_json(OBJ_CONFIG)
        if not self._from_bullet:
            self._configs = read_action_csv(get_csv_path(self.mode, self._state_id, self._step_id, self._table_id))
        else:
            self._configs = read_action_csv(DEFAULT_CONF_PATH)
        self.instruction = None
        self._load_instruction()
        self._transform_coord()
        self._width = width
        self._height = height
        self.config_obj = {}
        self.filepath = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '/'
        self.filename = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '.bullet'
        '''
        kuka
        
        '''
        self.maxVelocity = .35
        self.maxForce = 200.
        self.fingerAForce = 2
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 1
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 7

        '''
        UR5
        '''
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.obj_names = {}
        # if self._renders:
        #     self.cid = pb.connect(pb.SHARED_MEMORY)
        #     if (self.cid < 0):
        #         self.cid = pb.connect(pb.GUI)
        # else:
        #     self.cid = pb.connect(pb.DIRECT)
        self.client = BulletClient(pb.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        # RenderingPlugin(self.client, P3dRenderer(multisamples=64))
        # RenderingPlugin(self.client, PyrRenderer(platform='egl'))
        # egl = pkgutil.get_loader('eglRenderer')
        # self.client.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        self._set_observation_param()
        self._set_render_param()
        # self.setup()
        self.reset()
        self._dummy_run()

        # print("resetted")
        # render_img = self.get_render_img()
        # obs, depth = self.get_observation()
        # print("observation getted")

        # img = Image.fromarray(obs[:, :, :3])
        # img.save(str(self._state_id)+'_'+ str(self._table_id) +'_'+ str(self._step_id) + 'obs.png')
        # img = Image.fromarray(render_img[:, :, :3])
        # img.save(str(self._state_id)+'_'+ str(self._table_id) +'_'+ str(self._step_id) + 'render.png')
        # img = Image.fromarray(((depth / depth.max()) * 255).astype(np.uint8))
        # img.save(f'depth.png')
        # self._save_world_to_bullet()
        # conf = self._save_obj_config()
        # print(conf)
        # dump_json(conf, 'conf.json')
        # start = timer()
        # for _ in range(1000):
        #     self.get_observation()
        # end = timer()
        # value = 1000 / (end - start)
        # print("fps: {}".format(value))
    
    def _transform_coord(self):
        for ele in self._configs:
            pos, orn = transform(ele, self._configs[ele])
            self._obj_args[ele]['basePosition'] = pos
            self._obj_args[ele]['baseOrientation'] = orn
        self._obj_args['Coffee Cup BB']['basePosition'] = self._obj_args['Coffee Plate BB']['basePosition']
        self._obj_args['Coffee Cup BB']['basePosition'][2] = TABLE_HEIGHT_PLUS
        self._obj_args['Coffee Cup BB']['baseOrientation'] = pb.getQuaternionFromEuler([math.pi/2., 0, -math.pi/6.])
    
    def _set_observation_param(self):
        '''
        the observation for robot decision making and learnings
        '''
        self._view_matrix = pb.computeViewMatrixFromYawPitchRoll(
                                LOOK, DISTANCE, YAW, PITCH, ROLL, 2)
        fov = 30.
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = pb.computeProjectionMatrixFOV(fov, aspect, near, far)
    
    def _set_render_param(self):
        '''
        set render camera parameters
        '''
        self._render_view_matrix = pb.computeViewMatrixFromYawPitchRoll(
                RENDER_LOOK, RENDER_DISTANCE, RENDER_YAW, RENDER_PITCH, RENDER_ROLL, 2)
        fov = 30.
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._render_proj_matrix = pb.computeProjectionMatrixFOV(fov, aspect, near, far)
    
    def seed(self):
        '''
        random seeding for scene loading
        '''
        pass

    def _reset(self):
        self.robot_id = self.client.loadURDF(
            fileName=UR5_URDF,
            basePosition=(-0.5,0.6,1.07236), 
            baseOrientation=pb.getQuaternionFromEuler([0.0,0.,-math.pi/2.]), 
            useFixedBase=1,
            globalScaling=1.3
        )
        self.ee = Suction(currentdir, self.robot_id, 9, self.obj_ids)
        self.ee_tip = 10

        n_joints = self.client.getNumJoints(self.robot_id)
        joints = [self.client.getJointInfo(self.robot_id, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == self.client.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            self.client.resetJointState(self.robot_id, self.joints[i], self.homej[i])
        self.ee.release()
        self._load_world_from_config()
        self.client.stepSimulation()

    def reset(self):
        '''
        reset the envs
        '''
        self.client.resetSimulation()
        self.client.setTimeStep(self._time_step)
        self.client.setGravity(0, 0, -9.8)

        # floorid = self.client.loadURDF("plane.urdf")
        # # textureId = self.client.loadTexture(FLOOR_TEX)
        # # self.client.changeVisualShape(floorid, -1, textureUniqueId=textureId)
        
        # load robot
        # self.robot_id = self.client.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]

        # self.client.resetBasePositionAndOrientation(self.robot_id, 
        #                 posObj=(0.1,0.8,1.07236),
        #                 ornObj=pb.getQuaternionFromEuler([0.0,0.,-math.pi/2]))
        # self.endEffectorPos = [0.537, 0.0, 0.5]
        self._reset()
        # self.robot_id = self.client.loadURDF(
        #     fileName=UR5_URDF,
        #     basePosition=(-0.5,0.6,1.07236), 
        #     baseOrientation=pb.getQuaternionFromEuler([0.0,0.,-math.pi/2.]), 
        #     useFixedBase=1,
        #     globalScaling=1.3
        # )
        # self.ee = Suction(currentdir, self.robot_id, 9, self.obj_ids)
        # self.ee_tip = 10

        # n_joints = self.client.getNumJoints(self.robot_id)
        # joints = [self.client.getJointInfo(self.robot_id, i) for i in range(n_joints)]
        # self.joints = [j[0] for j in joints if j[2] == self.client.JOINT_REVOLUTE]

        # # Move robot to home joint configuration.
        # for i in range(len(self.joints)):
        #     self.client.resetJointState(self.robot_id, self.joints[i], self.homej[i])
        # self.ee.release()

        if self._from_bullet:
            self._load_world_from_bullet()
        self.client.stepSimulation()

    
    def setup(self):
        '''
        set up the envs
        '''
        self.client.resetSimulation()
        self.client.setTimeStep(self._time_step)
        self.client.setGravity(0, 0, -9.8)
        # floorid = self.client.loadURDF("plane.urdf")
        # # textureId = self.client.loadTexture(FLOOR_TEX)
        # # self.client.changeVisualShape(floorid, -1, textureUniqueId=textureId)
        
        # load robot
        # self.robot_id = self.client.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]

        # self.client.resetBasePositionAndOrientation(self.robot_id, 
        #                 posObj=(0.1,0.8,1.07236),
        #                 ornObj=pb.getQuaternionFromEuler([0.0,0.,-math.pi/2]))
        # self.endEffectorPos = [0.537, 0.0, 0.5]
        self.robot_id = self.client.loadURDF(
            fileName=UR5_URDF,
            basePosition=(-0.5,0.6,1.07236), 
            baseOrientation=pb.getQuaternionFromEuler([0.0,0.,-math.pi/2.]), 
            useFixedBase=1,
            globalScaling=1.3
        )
        self.ee = Suction(currentdir, self.robot_id, 9, self.obj_ids)
        self.ee_tip = 10

        n_joints = self.client.getNumJoints(self.robot_id)
        joints = [self.client.getJointInfo(self.robot_id, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == self.client.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            self.client.resetJointState(self.robot_id, self.joints[i], self.homej[i])
        self.ee.release()

        self.table_id, args = self._load_object(
            scale_factor=0.013,
            texture=TABLE_TEX,
            shapeType=pb.GEOM_MESH,
            fileName=TABLE_OBJ, 
            rgbaColor=None,
            baseMass=0,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2]),
            useFixedBase=1
        )
        self.config_obj['Table'] = args
        # v_cloth = self.client.createVisualShape(self.client.GEOM_BOX, halfExtents=[0.24, 0.3, 0.005], rgbaColor=[189/255, 195/255, 199/255, 1],specularColor=[0.4, .4, 0.8])
        # cloth = self.client.createCollisionShape(self.client.GEOM_BOX, halfExtents=[0.24, 0.3, 0.005])
        # self.cloth = self.client.createMultiBody(baseMass=1, baseCollisionShapeIndex=cloth, baseVisualShapeIndex=v_cloth, basePosition=[0.2, 0.2, 1.08])
        # self.obj_ids['fixed'].append(self.table_id)
        # self.obj_names['Coffee Plate BB'] = self.plate4
        self.cloth, args = self._load_object(baseMass=1.0, 
                                    fileName=TABLE_CLOTH,
                                    shapeType=pb.GEOM_MESH,
                                    texture=CLOTH_TEX,
                                    # halfExtents=[0.21, 0.28, 0.005],
                                    rgbaColor=[189/255, 195/255, 199/255, 1],
                                    basePosition=[0.34, -0.14, 1.08],
                                    baseOrientation=pb.getQuaternionFromEuler([0., 0, math.pi/2]),
                                    scale_factor=1.3
                                    )
        self.obj_ids['rigid'].append(self.cloth)
        self.obj_names['Table Cloth Sides'] = self.cloth
        self.config_obj['Table Cloth Sides'] = args
        self.wine_glass1, args = self._load_object(scale_factor=0.7, 
                                            baseMass=1.0,
                                            shapeType=pb.GEOM_MESH,
                                            fileName=WINE_GLASS,
                                            rgbaColor=[189/255, 195/255, 199/255, 1],
                                            basePosition=[0.5, 0.24, 1.08],
                                            baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                            )
        self.obj_ids['rigid'].append(self.wine_glass1)
        self.obj_names['Wine Glass BB'] = self.wine_glass1
        self.config_obj['Wine Glass BB'] = args
        self.wine_glass2, args = self._load_object(scale_factor=0.7, 
                                            baseMass=1.0,
                                            shapeType=pb.GEOM_MESH,
                                            fileName=WINE_GLASS,
                                            rgbaColor=[189/255, 195/255, 199/255, 1],
                                            basePosition=[0.5, 0.34, 1.08],
                                            baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                            )
        self.obj_ids['rigid'].append(self.wine_glass2)
        self.obj_names['Wine Glass BB (2)'] = self.wine_glass2
        self.config_obj['Wine Glass BB (2)'] = args
        self.wine_glass3, args = self._load_object(scale_factor=0.7, 
                                            baseMass=1.0,
                                            shapeType=pb.GEOM_MESH,
                                            fileName=WINE_GLASS,
                                            rgbaColor=[189/255, 195/255, 199/255, 1],
                                            basePosition=[0.5, 0.44, 1.08],
                                            baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                            )
        self.obj_ids['rigid'].append(self.wine_glass3)
        self.obj_names['Wine Glass BB (1)'] = self.wine_glass3
        self.config_obj['Wine Glass BB (1)'] = args
        self.water_glass, args = self._load_object(scale_factor=0.7, 
                                            baseMass=1.0,
                                            shapeType=pb.GEOM_MESH,
                                            fileName=WATER_GLASS,
                                            rgbaColor=[189/255, 195/255, 199/255, 1],
                                            basePosition=[-0.15, 0.5, 1.12],
                                            baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                            )
        self.obj_ids['rigid'].append(self.water_glass)
        self.obj_names['Water Glass BB'] = self.water_glass
        self.config_obj['Water Glass BB'] = args
        self.knife1, args = self._load_object(scale_factor=1.5, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=DINNER_KNIFE,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.08, 0.4, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.knife1)
        self.obj_names['Dinner Knife BB'] = self.knife1
        self.config_obj['Dinner Knife BB'] = args
        self.knife2, args = self._load_object(scale_factor=1.0, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=DINNER_KNIFE,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.08, 0.35, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.knife2)
        self.obj_names['Salad Knife BB'] = self.knife2
        self.config_obj['Salad Knife BB'] = args
        self.knife3, args = self._load_object(scale_factor=.7, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=DINNER_KNIFE,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.06, 0.3, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.knife3)
        self.obj_names['Butter Knife BB'] = self.knife3
        self.config_obj['Butter Knife BB'] = args
        self.spoon, args = self._load_object(scale_factor=.8, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=SPOON,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.06, -0.01, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.spoon)
        self.obj_names['Dessert Spoon BB'] = self.spoon
        self.config_obj['Dessert Spoon BB'] = args
        self.spoon2, args = self._load_object(scale_factor=1.2, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=SPOON,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.08, -0.08, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.spoon2)
        self.obj_names['Soup Spoon BB'] = self.spoon2
        self.config_obj['Soup Spoon BB'] = args
        self.fork1, args = self._load_object(scale_factor=1.4, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=DINNER_FORK,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.07, 0.2, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.fork1)
        self.obj_names['Dinner Fork BB'] = self.fork1
        self.config_obj['Dinner Fork BB'] = args
        self.fork2, args = self._load_object(scale_factor=1.0, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=DINNER_FORK,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.04, 0.15, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.fork2)
        self.obj_names['Salad Fork BB'] = self.fork2
        self.config_obj['Salad Fork BB'] = args
        self.fork3, args = self._load_object(scale_factor=.8, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=DINNER_FORK,
                                        rgbaColor=IRON_RGBA,
                                        basePosition=[-0.06, 0.05, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.fork3)
        self.obj_names['Dessert Fork BB'] = self.fork3
        self.config_obj['Dessert Fork BB'] = args
        self.plate1, args = self._load_object(scale_factor=.8, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=PLATE,
                                        colli_obj=PLATE_COL,
                                        rgbaColor=GOLD_RGBA,
                                        basePosition=[-0.02, -0.28, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([0, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.plate1)
        self.obj_names['Bread Plate BB'] = self.plate1
        self.config_obj['Bread Plate BB'] = args
        self.plate2, args = self._load_object(scale_factor=1.5, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=PLATE,
                                        colli_obj=PLATE_COL,
                                        rgbaColor=[189/255, 195/255, 199/255., 1],
                                        basePosition=[0.28, 0.34, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([0, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.plate2)
        self.obj_names['Charger BB'] = self.plate2
        self.config_obj['Charger BB'] = args
        self.plate3, args = self._load_object(scale_factor=1.2, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=PLATE,
                                        colli_obj=PLATE_COL,
                                        rgbaColor=[189/255, 195/255, 199/255., 1],
                                        basePosition=[-0.25, -0.28, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([0, 0, math.pi/2])
                                        )
        self.obj_ids['rigid'].append(self.plate3)
        self.obj_names['Dinner Plate BB'] = self.plate3
        self.config_obj['Dinner Plate BB'] = args
        self.napkin, args = self._load_object(scale_factor=1.0, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=NAPKIN,
                                        colli_obj=NAPKIN,
                                        rgbaColor=[189/255, 195/255, 199/255., 1],
                                        basePosition=[-0.33, 0.05, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([0, 0, math.pi])
                                        )
        self.obj_ids['rigid'].append(self.napkin)
        self.obj_names['Napkin Cloth BB'] = self.napkin
        self.config_obj['Napkin Cloth BB'] = args
        self.coffee_cup, args = self._load_object(scale_factor=1.2, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=COFFIE,
                                        colli_obj=COFFIE,
                                        rgbaColor=[189/255, 195/255, 199/255., 1],
                                        basePosition=[-0.33, 0.3, 1.10],
                                        baseOrientation=pb.getQuaternionFromEuler([math.pi/2., 0, -math.pi/6.])
                                        )
        self.obj_ids['rigid'].append(self.coffee_cup)
        self.obj_names['Coffee Cup BB'] = self.coffee_cup
        self.config_obj['Coffee Cup BB'] = args
        self.plate4, args = self._load_object(scale_factor=.8, 
                                        baseMass=1.0,
                                        shapeType=pb.GEOM_MESH,
                                        fileName=PLATE,
                                        colli_obj=PLATE_COL,
                                        rgbaColor=SILVER_RGBA,
                                        basePosition=[-0.33, 0.3, 1.08],
                                        baseOrientation=pb.getQuaternionFromEuler([0, 0, math.pi/2])
                                        )
        
        self.obj_ids['rigid'].append(self.plate4)
        self.obj_names['Coffee Plate BB'] = self.plate4
        self.config_obj['Coffee Plate BB'] = args
        self._exam_args()
        self.client.stepSimulation()
        # self._load_world()
    
    def _dummy_run(self):
        for _ in range(1000):
            self.client.stepSimulation()
    
    def _exam_args(self):
        for ele in self.config_obj:
            if bool(self.config_obj[ele]['kwargs']):
                for key in self.config_obj[ele]['kwargs']:
                    self.config_obj[ele][key] = self.config_obj[ele]['kwargs'][key]
                del self.config_obj[ele]['kwargs']
            if ele != 'Table':
                del self.config_obj[ele]['basePosition']
                del self.config_obj[ele]['baseOrientation']
            if 'useFixedBase' in self.config_obj[ele]:
                if self.config_obj[ele]['useFixedBase'] == 1:
                    self.config_obj[ele]['category'] = 'fixed'
                else:
                    self.config_obj[ele]['category'] = 'rigid'
            else:
                self.config_obj[ele]['category'] = 'rigid'
        dump_json(self.config_obj, '/home/zirui/tabletop_gym/tabletop_gym/envs/config/Objects.json')
    
    def _load_object(self, baseMass, fileName, shapeType, basePosition, baseOrientation, rgbaColor,
                    scale_factor=0.013, texture=None, colli_obj=None, 
                    **kwargs):
        '''
        load object to the physical world
        '''
        args = locals()
        del args['self']
        # print(args)
        if fileName is not None:
            visualShapeId = self.client.createVisualShape(
                shapeType=shapeType,
                fileName=fileName,
                rgbaColor=rgbaColor,
                specularColor=[0.4, .4, 0],
                meshScale=[scale_factor, scale_factor, scale_factor],)
            if colli_obj is not None:
                collisionShapeId = self.client.createCollisionShape(
                    shapeType=shapeType,
                    fileName=colli_obj,
                    meshScale=[scale_factor, scale_factor, scale_factor])
            else:
                collisionShapeId = self.client.createCollisionShape(
                    shapeType=shapeType,
                    fileName=fileName,
                    meshScale=[scale_factor, scale_factor, scale_factor])
        else:
            visualShapeId = self.client.createVisualShape(
                shapeType=shapeType,
                rgbaColor=rgbaColor,
                specularColor=[0.4, .4, 0],
                meshScale=[scale_factor, scale_factor, scale_factor],
                **kwargs)
            collisionShapeId = self.client.createCollisionShape(
                shapeType=shapeType,
                meshScale=[scale_factor, scale_factor, scale_factor],
                **kwargs)

        multiBodyId = self.client.createMultiBody(
            baseMass=baseMass,
            baseCollisionShapeIndex=collisionShapeId, 
            baseVisualShapeIndex=visualShapeId,
            basePosition=basePosition,
            baseOrientation=baseOrientation)
        if texture is not None:
            textureId = self.client.loadTexture(texture)
            self.client.changeVisualShape(multiBodyId, -1, textureUniqueId=textureId)
        return multiBodyId, args

    def _load_world_from_config(self):
        '''
        load all the objects to the physical engine according to the json config file
        '''
        # TODO: attach names into the objects
        for ele in self._obj_args:
            # print(ele)
            cate = self._obj_args[ele].pop('category')
            if ele == 'Table Cloth Sides':
                self._obj_args[ele]['basePosition'][2]  -= 0.02
            else:
                self._obj_args[ele]['basePosition'][2] += 0.02
            obj_id = self._load_object(
                **self._obj_args[ele]
            )
            self.obj_ids[cate].append(obj_id)
            self.obj_names[ele] = obj_id
    
    def _load_world_from_bullet(self):
        pb.restoreState(fileName=self.filepath + 'scene.bullet')

    def _load_instruction(self):
        json_path = get_json_path(self.mode, self._state_id, self._step_id, self._table_id)
        data = read_json(json_path)
        if len(data['instructions']) <= self._step_id:
            return None
        else:
            self.instruction = data['instructions'][self._step_id]

    def save_world_to_bullet(self):
        # TODO: make dir
        if not os.path.isfile(self.filepath + 'scene.bullet'): 
            try:
                if not os.path.isdir(self.filepath):
                    os.mkdir(self.filepath)
                self.client.saveBullet(self.filepath + 'scene.bullet')
            except IOError as exc:
                raise IOError("%s: %s" % (self.filepath + 'scene.bullet', exc.strerror))
    


    def save_scene(self):
        # save the scene to the bullet file and target json
        if self.instruction is not None:
            # print('saving')
            self.save_world_to_bullet()
            target_path = get_csv_path(self.mode, self._state_id, \
                self._step_id + 1, self._table_id)
            target = read_action_csv(target_path)
            # target['instruction'] = self.instruction
            target_dict = dict()
            for ele in target:
                target_dict[ele] = dict()
                pos, orn = transform(ele, target[ele])
                target_dict[ele]['pos'] = pos
                target_dict[ele]['orn'] = orn.tolist()
            target_dict['instruction'] = self.instruction
            dump_json(target_dict, self.filepath + 'info.json')
        

    def _save_obj_config(self):
        configs = dict()
        for ele in self.obj_ids['rigid']:
            obj_config = dict()
            obj_config['name'] = self.id2name(ele)
            obj_config['pose'], orn = self.client.getBasePositionAndOrientation(ele)
            obj_config['orn'] = self.client.getEulerFromQuaternion(orn)
            obj_config['orn_enable'] = check_orn_enable(self.id2name(ele))
            configs[self.id2name(ele)] = obj_config
        return configs
    
    def id2name(self, id):
        all_keys = [key for key, value in self.obj_names.items() if value == id]
        assert len(all_keys) == 1, ('there should be only one name for object {}, \
            but {} names are found'.format(id, len(all_keys)))
        return all_keys[0]

    def name2id(self, name):
        return self.obj_names[name]

    def get_observation(self):
        '''
        get the observation for robot
        '''
        w, h, rgba, depth, mask = self.client.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._view_matrix,
                               projectionMatrix=self._proj_matrix)
        rgb = rgba
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        # print(np.shape(depth))
        # print(depth)
        return np_img_arr[:, :, :3], np.array(depth), self.instruction
    
    def get_render_img(self):
        '''
        get render image
        '''
        _, _, rgba, _, _ = self.client.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._render_view_matrix,
                               projectionMatrix=self._render_proj_matrix)
        rgb = rgba
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        # print(np.shape(depth))
        # print(depth)
        return np_img_arr[:, :, :3]

    def apply_action(self, action):
        pass

    def _movej(self, targj, speed=0.01, timeout=5):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [self.client.getJointState(self.robot_id, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            self.client.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.joints,
                controlMode=self.client.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            self.client.stepSimulation()
        print(f'Warning: _movej exceeded {timeout} second timeout. Skipping.')
        return True

    def _movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self._solve_ik(pose)
        return self._movej(targj, speed)

    def _solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = self.client.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints