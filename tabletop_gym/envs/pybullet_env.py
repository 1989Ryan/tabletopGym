from json import dump
import os, inspect
from utils import HEIGHT, LIGHT_DIRECTION, OBJ_CONFIG_10, OBJ_CONFIG_10_A, OBJ_CONFIG_4, WIDTH, OBJ_CONFIG_10_B
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

from planning_ompl import PbOMPL, PbOMPLRobot
import pybullet as pb
import pybullet_data
import numpy as np

from ur5_utils.gripper import Suction
from ur5_utils.pickplace import PickPlaceContinuous
from utils import DEFAULT_STATE, DEFAULT_STEP, DEFAULT_TABLE, DEFAULT_CONF_PATH, DISTANCE,\
    LOOK, YAW, PITCH, RENDER_DISTANCE, RENDER_LOOK, RENDER_PITCH, RENDER_ROLL, RENDER_YAW, ROLL,\
    UR5_URDF, SHORT_FLOOR, FOV, NEAR, FAR, ASPECT,\
    read_action_csv, read_json, get_csv_path, transform, dump_json, check_orn_enable,\
    get_json_path_from_scene, get_segmentation_mask_object_and_link_index, RealSenseD415
import math, time
from pybullet_utils.bullet_client import BulletClient

from pybullet_rendering import RenderingPlugin

from PIL import Image
from pybullet_rendering.render.pyrender import PyrRenderer, PyrViewer
from timeit import default_timer as timer
import copy
import imageio

class Tabletop_Sim:
    '''
    Tabletop gym pybullet physical environment.
    Load the physical environments according to the 
    config file for different scenarios.
    '''
    def __init__(self,
                from_bullet=True,
                table_id=DEFAULT_TABLE,
                state_id=DEFAULT_STATE,
                step_id=DEFAULT_STEP,
                mode='train',
                time_step=1./240.,
                use_gui=False,
                width=WIDTH,
                height=HEIGHT,
                save_imgs = False,
                obj_number=12,
                record_video=False,
                record_cfg = None,
                ):
        '''
        config::scene configurations file path
        '''
        self._save_imgs = save_imgs
        self._from_bullet = from_bullet
        self._state_id = state_id
        self._step_id = step_id
        self._table_id = table_id
        self.mode = mode
        self._use_gui = use_gui
        self._time_step = time_step
        self.object_number = obj_number
        self.record_video = record_video
        self.record_cfg = record_cfg
        if self.record_video:
            self.gui_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                            'gui.mp4'),
                                            fps=self.record_cfg['fps'],
                                            format='FFMPEG',
                                            codec='h264',
                                            )
            self.video_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                            self.record_cfg['video_name']),
                                            fps=self.record_cfg['fps'],
                                            format='FFMPEG',
                                            codec='h264',
                                            )
        else:
            self.video_writer = None
            self.gui_writer = None
        self.cliport_cam_config = RealSenseD415.CONFIG
        if obj_number ==4:
            self._obj_args = read_json(OBJ_CONFIG_4)
        elif obj_number == 10:
            self._obj_args = read_json(OBJ_CONFIG_10)
        else:
            self._obj_args = read_json(OBJ_CONFIG_10_B)
        self.primitive = PickPlaceContinuous()
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
        print(self.filepath)
        self.filename = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '.bullet'
        '''
        UR5
        '''
        self.joint_bounds = [np.pi, 2.3562, 34, 34, 34, 34]
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.obj_names = {}
        if self._use_gui:
            self.client = BulletClient(pb.GUI)
        else:
            self.client = BulletClient(pb.DIRECT)
            RenderingPlugin(self.client, PyrRenderer(platform='egl', render_mask=True, shadows=True))
        
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planner = None
        '''
        rendering
        '''
        # RenderingPlugin(self.client, P3dRenderer(multisamples=0))
        # egl = pkgutil.get_loader('eglRenderer')
        # self.client.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # RenderingPlugin(self.client, PyrViewer)
        # RenderingPlugin(self.client, PyrRenderer(platform='egl', render_mask=False, shadows=True))
        # self.viewer = PyrViewer()
        self._set_observation_param()
        self._set_render_param()
        self.reset()
        self._dummy_run()
  
    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(self.client.getBaseVelocity(i)[0])
            for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)
    
    
    def get_info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        # Some tasks create and remove zones, so ignore those IDs.
        # removed_ids = []
        # if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
        #         isinstance(self.task, tasks.names['bag-alone-open'])):
        #   removed_ids.append(self.task.zone_id)

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = self.client.getBasePositionAndOrientation(obj_id)
                dim = self.client.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info
    
    def get_state(self):
        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = self.client.getBasePositionAndOrientation(obj_id)
                # dim = self.client.getVisualShapeData(obj_id)[0][3]
                info[self.id2name(obj_id)] = (pos, rot)
        return info

    def test_fps(self):
        start = timer()
        for _ in range(1000):
            self.get_observation()
        end = timer()
        value = 1000 / (end - start)
        print("fps: {}".format(value))
    
    def _transform_coord(self):
        for ele in self._configs:
            if ele in self._obj_args:
                pos, orn = transform(ele, self._configs[ele])
                self._obj_args[ele]['basePosition'] = pos
                self._obj_args[ele]['basePosition'][2] += 0.02
                self._obj_args[ele]['baseOrientation'] = orn
        if 'Coffee Cup BB' in self._obj_args and 'Coffee Plate BB' in self._obj_args:
            self._obj_args['Coffee Cup BB']['basePosition'] = copy.deepcopy(self._obj_args['Coffee Plate BB']['basePosition'])
            self._obj_args['Coffee Cup BB']['basePosition'][2] += 0.02
            self._obj_args['Coffee Cup BB']['baseOrientation'] = pb.getQuaternionFromEuler([math.pi/2., 0, -math.pi/6.])
        # self._obj_args['Dinner Plate BB']['basePosition'][2] -= 0.02

    def _set_observation_param(self):
        '''
        the observation for robot decision making and learnings
        '''
        self._view_matrix = pb.computeViewMatrixFromYawPitchRoll(
                                LOOK, DISTANCE, YAW, PITCH, ROLL, 2)
        fov = FOV
        aspect = ASPECT
        near = NEAR
        far = FAR
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
        if self._use_gui:
            # self.client.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, 1)
            self.client.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)
            self.client.resetDebugVisualizerCamera(cameraDistance=RENDER_DISTANCE/2, 
                                                cameraYaw=RENDER_YAW,
                                                cameraPitch=RENDER_PITCH,
                                                cameraTargetPosition=RENDER_LOOK)
    

    def _reset(self):
        # floorid = self.client.loadURDF("plane.urdf")
        # textureId = self.client.loadTexture(FLOOR_TEX)
        # self.client.changeVisualShape(floorid, -1, textureUniqueId=textureId)
        self.robot_id = self.client.loadURDF(
            fileName=UR5_URDF,
            basePosition=(-0.9,0.0,1.07236), 
            baseOrientation=pb.getQuaternionFromEuler([0.0,0.,0]), 
            useFixedBase=1,
            globalScaling=2.0
        )
        self.ee = Suction(currentdir, self.robot_id, 9, self.obj_ids, self.client)
        self.ee_tip = 10
        n_joints = self.client.getNumJoints(self.robot_id)
        joints = [self.client.getJointInfo(self.robot_id, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == self.client.JOINT_REVOLUTE]
        for i in range(len(self.joints)):
            self.client.resetJointState(self.robot_id, self.joints[i], self.homej[i])
        self.ee.release()
        if self.planner is None:
            self.ompl_robot = PbOMPLRobot(self.client, self.robot_id)
            self.joint_bounds = self.ompl_robot.joint_bounds
            self.planner = PbOMPL(self.client, self.ompl_robot)
            self.ll = self.ompl_robot.ll
            self.ul = self.ompl_robot.ul
            self.jr = self.ompl_robot.jr
        self.planner.obstacles.clear()
        self._load_world_from_config()
        self.client.stepSimulation()

    def reset(self):
        '''
        reset the envs
        '''
        self.client.resetSimulation()
        self.client.setTimeStep(self._time_step)
        self.client.setGravity(0, 0, -9.8)
        self._reset()
        self._set_observation_param()
        self._set_render_param()
        if self._from_bullet:
            self._load_world_from_bullet()
        floorid = self.client.loadURDF(SHORT_FLOOR, useFixedBase=1)
        # textureId = self.client.loadTexture(TABLE_TEX)
        # self.client.changeVisualShape(floorid, -1, textureUniqueId=textureId)
        self.client.stepSimulation()
        if self._from_bullet:
            for obj_ids in self.obj_ids.values():
                for obj_id in obj_ids:
                    self.planner.add_obstacles(obj_id)
        return self.get_observation()
    
    def reset_with_param(self, state_id, table_id,step_id, mode,\
                save_imgs=True, from_bullet=False, obj_number=12, record_video=False, record_cfg=None):
        self.record_video = record_video
        self.record_cfg = record_cfg
        if self.record_video:
            self.gui_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                            self.record_cfg['gui_name']),
                                            fps=self.record_cfg['fps'],
                                            format='FFMPEG',
                                            codec='h264',
                                            )
            self.video_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                            self.record_cfg['video_name']),
                                            fps=self.record_cfg['fps'],
                                            format='FFMPEG',
                                            codec='h264',
                                            )
        else:
            self.video_writer = None
            self.gui_writer = None
        self.object_number = obj_number
        if obj_number ==4:
            self._obj_args = read_json(OBJ_CONFIG_4)
        elif obj_number == 10:
            self._obj_args = read_json(OBJ_CONFIG_10)
        else:
            self._obj_args = read_json(OBJ_CONFIG_10_B)
        # print(self._obj_args)
        self._state_id = state_id
        self._table_id = table_id
        self._step_id = step_id
        self.mode = mode
        self._save_imgs = save_imgs
        self._from_bullet = from_bullet
        if not self._from_bullet:
            self._configs = read_action_csv(get_csv_path(self.mode, self._state_id, self._step_id, self._table_id))
        else:
            self._configs = read_action_csv(DEFAULT_CONF_PATH)
        self.instruction = None
        self._load_instruction()
        self._transform_coord()
        self.config_obj = {}
        self.filepath = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '/'
        # print(self.filepath)
        self.filename = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '.bullet'
        self.reset()
        self._dummy_run()
    
    def _dummy_run(self):
        for _ in range(100):
            self.client.stepSimulation()
    
    def run(self):
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
        dump_json(self.config_obj, 'tabletop_gym/tabletop_gym/envs/config/Objects.json')
    
    def _load_object(self, baseMass, fileName, shapeType, basePosition, baseOrientation, rgbaColor,
                    scale_factor=0.013, texture=None, colli_obj=None, 
                    **kwargs):
        '''
        load object to the physical world
        '''
        args = locals()
        del args['self']
        shift1 = [0, 0.1, 0]
        shift2 = [0, 0, 0]
        if fileName is not None:
            if shapeType == 3:
                # visualShapeId = self.client.createVisualShapeArray(
                #     shapeTypes=[3, 5],
                #     halfExtents=[kwargs['kwargs']['halfExtents'], [0, 0, 0]],
                #     fileNames=['', fileName],
                #     rgbaColors=[None, rgbaColor],
                #     # specularColors=[[0.4, .4, 0], [0, 0, 0]],
                #     visualFramePositions=[
                #                              shift1,
                #                              shift2,
                #                          ],
                #     meshScales=[[scale_factor, scale_factor, scale_factor], [1, 1, 1]],
                # )
                visualShapeId = self.client.createVisualShape(
                    shapeType=5,
                    fileName=fileName,
                    rgbaColor=rgbaColor,
                    specularColor=[0.4, .4, 0],
                    meshScale=[scale_factor, scale_factor, scale_factor],)
            
            else:
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
                # collisionShapeId = self.client.createCollisionShape(
                # shapeType=pb.GEOM_BOX,
                # halfExtents=[0.5, 0.5, 0.5],
                # # meshScale=[scale_factor, scale_factor, scale_factor],
                # # **kwargs
                # )
                if shapeType == 3:
                    collisionShapeId = self.client.createCollisionShape(
                    shapeType=pb.GEOM_BOX,
                    halfExtents=kwargs['kwargs']['halfExtents'],
                    collisionFramePosition=shift1,
                    # halfExkwargs
                    )
                    # collisionShapeId = self.client.createCollisionShapeArray(
                    #     shapeTypes=[3, 5],
                    #     halfExtents=[kwargs['kwargs']['halfExtents'], [0, 0, 0]],
                    #     fileNames=['', fileName],
                    #     collisionFramePositions=[
                    #                             shift1,
                    #                             shift2,
                    #                         ],
                    #     meshScales=[[scale_factor, scale_factor, scale_factor], [1, 1, 1]],
                    #     )
                else:
                    collisionShapeId = self.client.createCollisionShape(
                        shapeType=5,
                        fileName=fileName,
                        meshScale=[scale_factor, scale_factor, scale_factor]
                    )
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
        
        for ele in self._obj_args:
            cate = self._obj_args[ele].pop('category')
            if ele == 'Table Cloth Sides':
                self._obj_args[ele]['basePosition'][2]  -= 0.02
            elif 'Wine Glass' in ele:
                self._obj_args[ele]['basePosition'][2] = 1.17
            else:
                self._obj_args[ele]['basePosition'][2] += 0.02
            obj_id, _ = self._load_object(
                **self._obj_args[ele]
            )
            self.obj_ids[cate].append(obj_id)
            self.obj_names[ele] = obj_id
        # print(self.obj_names)
    
    def _load_world_from_bullet(self):
        print(self.filepath)
        self.client.restoreState(fileName=self.filepath + 'scene.bullet')

    def _load_instruction(self):
        json_path = get_json_path_from_scene(self.mode, self._state_id, self._step_id, self._table_id)
        if os.path.isfile(json_path):
            data = read_json(json_path)
            self.instruction = data['instruction']

    def save_world_to_bullet(self):
        
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
            if self._save_imgs:
                img, _, _ = self.get_observation()
                obsimg = Image.fromarray(img)
                obsimg.save(self.filepath + 'observation.png')
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
        w, h, rgba, depth, mask = self.client.getCameraImage(
                                # shadow=1,
                                # lightDirection=LIGHT_DIRECTION,
                                width=self._width,
                                height=self._height,
                                viewMatrix=self._view_matrix,
                                projectionMatrix=self._proj_matrix)
        rgb = rgba
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        # print(np.shape(depth))
        # print(depth)
        ee_sensor = self.ee.detect_contact()
        # Get depth image.
        # print(np.shape(np.array(depth)))
        depth_image_size = (WIDTH, HEIGHT)
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (FAR + NEAR - (2. * zbuffer - 1.) * (FAR - NEAR))
        depth = (2. * NEAR * FAR) / depth
        return np_img_arr[:, :, :3], np.array(depth), self.instruction, ee_sensor
    
    def get_observation_with_mask(self):
        '''
        get the observation for robot with mask
        '''
        w, h, rgba, depth, mask = self.client.getCameraImage(
                                # shadow=1,
                                width=self._width,
                                height=self._height,
                                viewMatrix=self._view_matrix,
                                projectionMatrix=self._proj_matrix,
                                # flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                # renderer=pb.ER_TINY_RENDERER,
                                )
        # print(np.shape(mask))
        rgb = rgba
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        # mask = np.reshape(mask, (h, w))
        # print(np.shape(depth))
        # print(depth)
        ee_sensor = self.ee.detect_contact()
        # Get depth image.
        # print(np.shape(np.array(depth)))
        depth_image_size = (WIDTH, HEIGHT)
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (FAR + NEAR - (2. * zbuffer - 1.) * (FAR - NEAR))
        depth = (2. * NEAR * FAR) / depth
        return np_img_arr[:, :, :3], np.array(depth), mask, self.instruction, ee_sensor

    def get_cliport_img(self):
        obs = {'color': (), 'depth': ()}
        masks = ()
        for config in self.cliport_cam_config:
            color, depth, mask = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)
            masks += (mask,)
        return obs, masks

    def render_camera(self, config, image_size=None, shadow=1):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config['image_size']

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = pb.computeViewMatrix(config['position'], lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self.client.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            # shadow=shadow,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # renderer=pb.ER_BULLET_HARDWARE_OPENGL
            )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        
        # depth_image_size = (image_size[0], image_size[1])
        # zbuffer = np.array(depth).reshape(depth_image_size)
        # depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        # depth = (2. * znear * zfar) / depth
        # if config['noise']:
        #     depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        # segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

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
    
    def get_obj_pose(self, id):
        return self.client.getBasePositionAndOrientation(id)

    #########################################################
    # Task completion info
    #########################################################

    def done(self):
        '''
        TODO: check whether the task is done
        '''
        return False

    def reward(self):
        '''
        TODO: return the reward of the task
        '''
        return 0.0, self.get_info()


    #########################################################
    # Robot Info
    #########################################################

    def is_not_fixed(self, joint_idx):
        '''
        check whether the joint is fixed
        args
            joint_idx: joint index, defined in urdf file.
        '''
        joint_info = self.client.getJointInfo(self.robot_id, joint_idx)
        return joint_info[2] != self.client.JOINT_FIXED
    
    def apply_action(self, action):
        timeout = self.primitive(self._movej, self._movep, self.ee, action)
        return timeout

    # def _movej__(self, targj, speed=0.01, timeout=5):
    #     
    #     """Move UR5 to target joint configuration."""
    #     # print("haha")
    #     t0 = time.time()
    #     while (time.time() - t0) < timeout:
    #         currj = [self.client.getJointState(self.robot_id, i)[0] for i in self.joints]
    #         currj = np.array(currj)
    #         diffj = targj - currj
    #         if all(np.abs(diffj) < 1e-2):
    #             return False

    #         # Move with constant velocity
    #         norm = np.linalg.norm(diffj)
    #         v = diffj / norm if norm > 0 else 0
    #         stepj = currj + v * speed
    #         gains = np.ones(len(self.joints))
    #         self.client.setJointMotorControlArray(
    #             bodyIndex=self.robot_id,
    #             jointIndices=self.joints,
    #             controlMode=self.client.POSITION_CONTROL,
    #             targetPositions=stepj,
    #             positionGains=gains)
    #         self.client.stepSimulation()
    #     print(f'Warning: _movej exceeded {timeout} second timeout. Skipping.')
    #     return True

    def _movej(self, targj, speed=0.01, timeout=5):
        '''
        joint motion planning by ompl planning lib. 
        args:
            targj: target joint pose
            speed: speed of the motion
            timeout: maximum time for finding the planning path
        '''
        # print(targj)
        currj = [self.client.getJointState(self.robot_id, i)[0] for i in self.joints]
        if self._use_gui:
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_RENDERING, 0)
        res, path = self.planner.plan_start_goal(start=currj, goal=targj)
        if self._use_gui:
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_RENDERING, 1)
        if res:
            self.planner.execute(path, dynamics=False, video=self.record_video, video_writer=self.video_writer, 
                get_obs=self.get_observation_with_mask, gui_writer=self.gui_writer, get_gui_obs=self.get_render_img)
            return False
        else:
            print(f'Warning: _movej exceeded {timeout} second timeout. Skipping.')
        return True
        # pass

    def _movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        pos, orn = pose
        # orn = [0.,0.,0.,1.]
        pose = (pos, orn)
        targj = self._solve_ik(pose)
        # print("goal: {}".format(targj))
        return self._movej(targj, speed)

    def _solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = self.client.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            # lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            # upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            # jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints
    
    def get_ee_pose(self):
        return self.client.getLinkState(self.robot_id, self.ee_tip)[0:2]

    def collision_checking(self):
        '''
        
        '''
        pass
    
    def start_video(self):
        self.gui_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                            'gui.mp4'),
                                            fps=self.record_cfg['fps'],
                                            format='FFMPEG',
                                            codec='h264',
                                            )
        self.video_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                            self.record_cfg['video_name']),
                                            fps=self.record_cfg['fps'],
                                            format='FFMPEG',
                                            codec='h264',
                                            )

    def close_video(self):
        self.video_writer.close()
        self.gui_writer.close()

    # def __del__(self):
    #     pb.disconnect(physicsClientId=self.client._client)

    ###########################################
    # PDDLStream used functions
    ###########################################

