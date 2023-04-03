from json import dump
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
from utils import HEIGHT, LIGHT_DIRECTION, OBJ_CONFIG_10, OBJ_CONFIG_10_A, OBJ_CONFIG_4, WIDTH, OBJ_CONFIG_10_B
# print("current_dir=" + currentdir)
import random
# from planning_ompl import PbOMPL, PbOMPLRobot
import pybullet as pb
import pybullet_data
import numpy as np
import pickle

from ur5_utils.gripper import Suction
from ur5_utils.pickplace import PickPlaceContinuous
from utils import DEFAULT_STATE, DEFAULT_STEP, DEFAULT_TABLE, DEFAULT_CONF_PATH, DISTANCE,\
    LOOK, YAW, PITCH, RENDER_DISTANCE, RENDER_LOOK, RENDER_PITCH, RENDER_ROLL, RENDER_YAW, ROLL,\
    UR5_URDF, SHORT_FLOOR, FOV, NEAR, FAR, ASPECT, OBJECT_OBJ_MAT_CONF, ENV_CONFIG, INTERPOLATE_NUM,\
    read_action_csv, read_json, get_csv_path, transform, dump_json, check_orn_enable,\
    get_json_path_from_scene, get_segmentation_mask_object_and_link_index, RealSenseD415, camera_coord, OBJ_CONFIG,\
    material_weights, color_weights, colors_name_train, colors_name_test_1, colors_name_test_2,\
    colors, render_camera_coord, getQuaternionFromMatrix, inv_transform
import math, time
from pybullet_utils.bullet_client import BulletClient


from PIL import Image
from timeit import default_timer as timer
import copy
import imageio

import nvisii

class Tabletop_Sim:
    '''
    Tabletop gym pybullet physical environment.
    Load the physical environments according to the 
    config file for different scenarios.
    '''
    def __init__(self,
                from_bullet=False,
                table_id=DEFAULT_TABLE,
                state_id=DEFAULT_STATE,
                step_id=DEFAULT_STEP,
                mode='test',
                unseen=False,
                time_step=1./240.,
                use_gui=False,
                width=WIDTH,
                height=HEIGHT,
                save_imgs = False,
                obj_number=11,
                record_video=False,
                record_cfg = None,
                indivisual_loading=False,
                ):
        '''
        config::scene configurations file path
        '''
        # TODO: loads all the materials and obj files in the beginning
        self.texture_name = {}
        self.indivisual_loading = indivisual_loading
        self.obj_list_json = None
        self.mesh_type = {}
        self.mentioned_objects = []
        self.mentioned_objects_ref = []
        self.grid = np.zeros((32, 32))
        self.name_id_dict = {}
        nvisii.initialize(headless=True, lazy_updates=True)
        nvisii.configure_denoiser(use_albedo_guide=True, use_normal_guide=True, use_kernel_prediction=True)
        self.load_materials()
        nvisii.enable_denoiser()
        if obj_number is not None:
            self.human_annotate = False
        else:
            self.human_annotate = True
        self.obj_bb2ref_dict = read_json("tabletop_gym/envs/config/ref2id2.json")
        self.bb2ref = read_json("tabletop_gym/envs/config/bb2ref.json")
        # Create a camera
        
        # camera = nvisii.entity.create(
        #     name = "camera",
        #     transform = nvisii.transform.create("camera"),
        #     camera = nvisii.camera.create_from_fov(
        #         name = "camera", 
        #         field_of_view = FOV/180 * np.pi,
        #         aspect = 1.0
        #     )
        # )
        # camera.get_transform().look_at(
        #     at = (0.08, 0.02, 1.07),
        #     up = (0,0,1),
        #     eye = camera_coord(),
        # )
        self.config_nvisii_camera()

        # Change the dome light intensity
        nvisii.set_dome_light_intensity(1.0)

        # atmospheric thickness makes the sky go orange, almost like a sunset
        nvisii.set_dome_light_sky(sun_position=(5,5,5), atmosphere_thickness=1.0, saturation=1.0)
        self.sun = nvisii.entity.create(
            name = "sun",
            mesh = nvisii.mesh.create_sphere("sphere"),
            transform = nvisii.transform.create("sun"),
            light = nvisii.light.create("sun")
        )
        self.moon = nvisii.entity.create(
            name = "moon",
            mesh = nvisii.mesh.create_sphere("sphere_2"),
            transform = nvisii.transform.create("moon"),
            light = nvisii.light.create("moon")
        )
        self.unseen = unseen
        self._save_imgs = save_imgs
        self._from_bullet = from_bullet
        self._state_id = state_id
        self._step_id = step_id
        self._table_id = table_id
        self.mode = mode
        if self.mode == 'test' and self.unseen:
            self.color_list = colors_name_test_2
        elif self.mode == 'test':
            self.color_list = colors_name_test_1
        else:
            self.color_list = colors_name_train 
        self._use_gui = use_gui
        self._time_step = time_step
        self.object_number = obj_number
        self.record_video = record_video
        self.record_cfg = record_cfg
        if self.record_video:
            self.video_frame_counter = 0
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
        
        self.obj_number = obj_number
        if self.obj_number == 4:
            self.objects_lists = read_json(OBJ_CONFIG_4)
        elif self.obj_number == 10:
            self.objects_lists = read_json(OBJ_CONFIG_10)
        else:
            self.objects_lists = read_json(OBJ_CONFIG_10_B)
        # elif obj_number == 10:
        #     self._obj_args = read_json(OBJ_CONFIG_10)
        # else:
        #     self._obj_args = read_json(OBJ_CONFIG_10_B)
        self.primitive = PickPlaceContinuous()
        if not self._from_bullet:
            self._configs = read_action_csv(get_csv_path(self.mode, self._state_id, self._step_id, self._table_id))
        else:
            self._configs = read_action_csv(DEFAULT_CONF_PATH)
        self.instruction = None
        self.obj_mesh_name = {}
        self._load_instruction()
        # self._transform_coord()
        self._width = width
        self._height = height
        self.config_obj = {}
        self.filepath = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '/'
        self.filename = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '.bullet'
        '''
        UR5
        '''
        self.joint_bounds = [np.pi, 2.3562, 34, 34, 34, 34]
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        if self._use_gui:
            self.client = BulletClient(pb.GUI)
        else:
            self.client = BulletClient(pb.DIRECT)
            # RenderingPlugin(self.client, PyrRenderer(platform='egl', render_mask=True, shadows=True))
        
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

    def reset_nvisii_camera(self):
        self.camera_nvisii_list = {}
        camera = nvisii.entity.create(
            name = "camera",
            transform = nvisii.transform.get("camera"),
            camera = nvisii.camera.get("camera")
        )
        self.camera_nvisii_list[0] = camera
        index = 1
        for config in RealSenseD415.CONFIG:
            camera = nvisii.entity.create(
                name = f"camera_{index}",
                transform = nvisii.transform.get(f"camera_{index}"),
                # camera = nvisii.camera.create_from_fov(
                #     name = f"camera_{index}", 
                #     field_of_view = fovh,
                #     aspect = aspect_ratio
                # )
                camera = nvisii.camera.get(f"camera_{index}")
            )
            # print(lookat)
            self.camera_nvisii_list[index] = camera
            index += 1
        camera = nvisii.entity.create(
            name = "camera_gui",
            transform = nvisii.transform.get("camera_gui"),
            camera = nvisii.camera.get("camera_gui")
        )

        self.camera_nvisii_list['gui'] = camera

    def update_visual_objects(self, object_ids, pkg_path, nv_objects=None):
        # object ids are in pybullet engine
        # pkg_path is for loading the object geometries
        # nv_objects refers to the already entities loaded, otherwise it is going 
        # to load the geometries and create entities. 
        if nv_objects is None:
            nv_objects = { }
        for object_id in object_ids:
            for idx, visual in enumerate(self.client.getVisualShapeData(object_id)):
                # Extract visual data from pybullet
                objectUniqueId = visual[0]
                linkIndex = visual[1]
                visualGeometryType = visual[2]
                dimensions = visual[3]
                meshAssetFileName = visual[4]
                local_visual_frame_position = visual[5]
                local_visual_frame_orientation = visual[6]
                rgbaColor = visual[7]
                
                world_link_frame_position = (0,0,0)
                world_link_frame_orientation = (0,0,0,1)
                if linkIndex == -1:
                    dynamics_info = self.client.getDynamicsInfo(object_id,-1)
                    inertial_frame_position = dynamics_info[3]
                    inertial_frame_orientation = dynamics_info[4]
                    base_state = self.client.getBasePositionAndOrientation(objectUniqueId)
                    world_link_frame_position = base_state[0]
                    world_link_frame_orientation = base_state[1]    
                    m1 = nvisii.translate(nvisii.mat4(1), nvisii.vec3(inertial_frame_position[0], inertial_frame_position[1], inertial_frame_position[2]))
                    m1 = m1 * nvisii.mat4_cast(nvisii.quat(inertial_frame_orientation[3], inertial_frame_orientation[0], inertial_frame_orientation[1], inertial_frame_orientation[2]))
                    m2 = nvisii.translate(nvisii.mat4(1), nvisii.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
                    m2 = m2 * nvisii.mat4_cast(nvisii.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))
                    m = nvisii.inverse(m1) * m2
                    q = nvisii.quat_cast(m)
                    world_link_frame_position = m[3]
                    world_link_frame_orientation = q
                else:
                    linkState = self.client.getLinkState(objectUniqueId, linkIndex)
                    world_link_frame_position = linkState[4]
                    world_link_frame_orientation = linkState[5]
                
                # Name to use for components
                object_name = str(objectUniqueId) + "_" + str(linkIndex)

                meshAssetFileName = meshAssetFileName.decode('UTF-8')
                if object_name not in nv_objects:
                    # Create mesh component if not yet made
                    if visualGeometryType == self.client.GEOM_MESH:
                        try:
                            nv_objects[object_name] = nvisii.import_scene(
                                meshAssetFileName
                            )
                        except Exception as e:
                            print(e)
                            pass
                
                if visualGeometryType != 5: continue

                if object_name not in nv_objects: continue

                # Link transform
                m1 = nvisii.translate(nvisii.mat4(1), nvisii.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
                m1 = m1 * nvisii.mat4_cast(nvisii.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))

                # Visual frame transform
                m2 = nvisii.translate(nvisii.mat4(1), nvisii.vec3(local_visual_frame_position[0], local_visual_frame_position[1], local_visual_frame_position[2]))
                m2 = m2 * nvisii.mat4_cast(nvisii.quat(local_visual_frame_orientation[3], local_visual_frame_orientation[0], local_visual_frame_orientation[1], local_visual_frame_orientation[2]))
                
                # Set root transform of visual objects collection to above transform
                nv_objects[object_name].transforms[0].set_transform(m1 * m2)
                nv_objects[object_name].transforms[0].set_scale(dimensions)

                for m in nv_objects[object_name].materials:
                    m.set_base_color((rgbaColor[0] ** 2.2, rgbaColor[1] ** 2.2, rgbaColor[2] ** 2.2))

                # todo... add support for spheres, cylinders, etc
                # print(visualGeometryType)
        return nv_objects

    def config_nvisii_camera(self):
        self.camera_nvisii_list = {}
        camera = nvisii.entity.create(
            name = "camera",
            transform = nvisii.transform.create("camera"),
            camera = nvisii.camera.create_from_fov(
                name = "camera", 
                field_of_view = FOV/180 * np.pi,
                aspect = 1.0
            )
        )
        camera.get_transform().look_at(
            at = (0.08, 0.02, 1.07),
            up = (0,0,1),
            eye = camera_coord(),
        )

        self.camera_nvisii_list[0] = camera
        index = 1
        for config in RealSenseD415.CONFIG:
            image_size = config["image_size"]
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
            fovh = np.arctan(fovh) * 2

            # Notes: 1) FOV is vertical FOV 2) aspect must be float
            aspect_ratio = image_size[1] / image_size[0]
            camera = nvisii.entity.create(
                name = f"camera_{index}",
                transform = nvisii.transform.create(f"camera_{index}"),
                # camera = nvisii.camera.create_from_fov(
                #     name = f"camera_{index}", 
                #     field_of_view = fovh,
                #     aspect = aspect_ratio
                # )
            camera = nvisii.camera.create_from_intrinsics(
                    name = f"camera_{index}", 
                    fx = config['intrinsics'][0],
                    fy = config['intrinsics'][4],
                    cx = config['intrinsics'][2],
                    cy = config['intrinsics'][5],
                    width = image_size[1],
                    height = image_size[0],
                    near = znear,
                    far = zfar
                )
            )
            # print(lookat)
            camera.get_transform().look_at(
                at = (lookat[0],lookat[1],lookat[2]),
                up = (0,0,1),
                eye = config["position"],
            )
            self.camera_nvisii_list[index] = camera
            index += 1
        camera = nvisii.entity.create(
            name = "camera_gui",
            transform = nvisii.transform.create("camera_gui"),
            camera = nvisii.camera.create_from_fov(
                name = "camera_gui", 
                field_of_view = FOV/180 * np.pi,
                aspect = 640/480
            )
        )
        camera.get_transform().look_at(
            at = RENDER_LOOK,
            up = (0,0,1),
            eye = render_camera_coord(),
        )

        self.camera_nvisii_list['gui'] = camera

    def load_materials(self):
        object_conf = read_json(OBJECT_OBJ_MAT_CONF)
        if self.indivisual_loading:
            self.object_conf = object_conf
        else:
            for ele in object_conf:
                if object_conf[ele] is not None:
                    for obj in object_conf[ele]:
                        mesh_path = object_conf[ele][obj]["meshes"]
                        texture_path = object_conf[ele][obj]["texture"]
                        self.texture_name[obj] = object_conf[ele][obj]["name"]
                        print("loading {}".format(object_conf[ele][obj]["name"]))
                        self.mesh_type[obj] = ele
                        nvisii.mesh.create_from_file(obj, mesh_path)
                        if texture_path is not None:
                            nvisii.texture.create_from_file(obj, texture_path)
    
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
                pos, rot = self.client.getBasePositionAndOrientation(obj_id + 3)
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
                pos, orn = transform(ele, self._configs[ele], self.obj_mesh_name[ele])
                self._obj_args[ele]['basePosition'] = pos
                self._obj_args[ele]['basePosition'][2] += 0.02
                self._obj_args[ele]['baseOrientation'] = orn
        if 'Napkin Cloth BB' in self._obj_args:
            self._obj_args['Napkin Cloth BB']['basePosition'][2] += 0.02
        if 'Dinner Plate BB' in self._obj_args:
            if 'Napkin Cloth BB' in self._obj_args:
                self._obj_args['Napkin Cloth BB']['basePosition'][2] += 0.02
            self._obj_args['Dinner Plate BB']['basePosition'][2] += 0.02
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
        
        # if self.planner is None:
        #     self.ompl_robot = PbOMPLRobot(self.client, self.robot_id)
        #     self.joint_bounds = self.ompl_robot.joint_bounds
        #     self.planner = PbOMPL(self.client, self.ompl_robot)
        #     self.ll = self.ompl_robot.ll
        #     self.ul = self.ompl_robot.ul
        #     self.jr = self.ompl_robot.jr
        # self.planner.obstacles.clear()
        self.ids_pybullet_and_nvisii_names = []
        self.id_transfer = {}
        self.id_transfer_2 = {}
        # self._load_world_from_config()
        self._dummy_run()
        # self.nv_objects = self.update_visual_objects([self.robot_id, self.ee.body, self.ee.base], '.')

    def reset(self):
        '''
        reset the envs
        '''
        self.grid = np.zeros((32, 32))
        self.client.resetSimulation()
        self.client.setTimeStep(self._time_step)
        self.client.setGravity(0, 0, -9.8)
        nvisii.entity.clear_all()
        self.reset_nvisii_camera()
        self.name_id_dict = {}
        self.obj_position = {}
        # Change the dome light intensity
        nvisii.set_dome_light_intensity(1.0)

        # atmospheric thickness makes the sky go orange, almost like a sunset
        nvisii.set_dome_light_sky(sun_position=(5,5,5), atmosphere_thickness=1.0, saturation=1.0)
        self.sun = nvisii.entity.create(
            name = "sun",
            mesh = nvisii.mesh.get("sphere"),
            transform = nvisii.transform.get("sun"),
            light = nvisii.light.get("sun")
        )
        self.moon = nvisii.entity.create(
            name = "moon",
            mesh = nvisii.mesh.get("sphere_2"),
            transform = nvisii.transform.get("moon"),
            light = nvisii.light.get("moon")
        )
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self._obj_args = read_json(ENV_CONFIG)
        self.obj_names = {}
        if not self.indivisual_loading:
            if self.obj_number ==4:
                r = random.choice([3, 4, 5])
            elif self.obj_number == 10:
                r = random.choice([7, 8, 9, 10])
            elif self.obj_number is not None:
                r = random.choice([7, 8, 9, 10])
            else:
                r = random.choice([5, 6, 7])
            if self.human_annotate:
                r = r - len(self.mentioned_objects)
                tmp_list = copy.deepcopy(self.objects_lists)
                object_selected = random.sample(
                    list(set(tmp_list.keys()) - set(self.mentioned_objects)),
                    k = r 
                )
                object_selected = object_selected + self.mentioned_objects
            else:
                object_selected = random.sample(
                    list(set(self.objects_lists.keys())), 
                    k = r 
                )
            if self.obj_list_json is None:
                for ele in object_selected:
                    self._obj_args[ele] = copy.deepcopy(self.objects_lists[ele])
            else:
                for ele in self.obj_list_json:
                    if ele in self.objects_lists:
                        self._obj_args[ele] = copy.deepcopy(self.objects_lists[ele])
       # print(list(self._obj_args.keys()))
        # print(self._obj_args)
        # Lets add a sun light
        self.object_property = {}
        theta_1 = random.uniform(0, 2 * np.pi)
        self.sun.get_transform().set_position((1 * math.cos(theta_1), 1 * math.sin(theta_1),8))
        self.sun.get_light().set_temperature(random.uniform(4000, 6000))
        self.sun.get_light().set_intensity(random.uniform(100, 110))

        theta_2 = random.uniform(0, 2 * np.pi)
        
        self.moon.get_transform().set_position((9 * math.cos(theta_2), 9 * math.sin(theta_2),3))
        self.moon.get_light().set_temperature(random.uniform(4000, 6000))
        self.moon.get_light().set_intensity(random.uniform(300, 400))


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
                save_imgs=True, from_bullet=False, obj_number=12, record_video=False, record_cfg=None, obj_list=None):
        
        self.record_video = record_video
        self.record_cfg = record_cfg
        if self.record_video:
            self.video_frame_counter = 0
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
        self.obj_number = obj_number
        self.object_number = obj_number
        if self.obj_number == 4:
            self.objects_lists = read_json(OBJ_CONFIG_4)
        elif self.obj_number == 10:
            self.objects_lists = read_json(OBJ_CONFIG_10)
        else:
            self.objects_lists = read_json(OBJ_CONFIG_10_B)
        self.idnamedict = read_json("tabletop_gym/envs/config/id2labelBB.json")
        # print(self._obj_args)
        self._state_id = state_id
        self._table_id = table_id
        self._step_id = step_id
        self.mode = mode
        self._save_imgs = save_imgs
        self._from_bullet = from_bullet
        self.obj_mesh_name = {}
        if not self._from_bullet:
            self._configs = read_action_csv(get_csv_path(self.mode, self._state_id, self._step_id, self._table_id))
        else:
            self._configs = read_action_csv(DEFAULT_CONF_PATH)
        self.instruction = None
        self._load_instruction()
        print(self.instruction)
        if self.obj_number is None:
            self.human_annotate = self.get_obj_from_ins()
        else:
            self.human_annotate = False
        # self._transform_coord()
        self.config_obj = {}
        self.filepath = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '/'
        # print(self.filepath)
        self.filename = currentdir + '/scenes/' + self.mode + '/' + str(self._state_id) + '_' + \
            str(self._table_id) + '_' + str(self._step_id) + '.bullet'
        if self.obj_number is None and not self.human_annotate:
            return
        self.obj_list_json = obj_list
        self.reset()
        self._dummy_run()
    
    def get_obj_from_ins(self):
        self.ins_obj = [self.obj_bb2ref_dict[ele] for ele in self.obj_bb2ref_dict if ele in self.instruction]
        self.mentioned_objects = []
        self.mentioned_objects_ref = []
        for ele in self.bb2ref:
            for ref in self.bb2ref[ele]:
                if ref in self.instruction.lower():
                    if ele not in self.mentioned_objects:
                        self.mentioned_objects.append(ele)
        if len(self.mentioned_objects) not in [2, 3]:
            return False
        else:
            return True
    
    def _dummy_run(self, step=None):
        if step is None:
            for _ in range(300):
                self.client.stepSimulation()
        else:
            for _ in range(step):
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
    
    def _load_object(self, ele, objectclass, objectname, baseMass, fileName, shapeType, rgbaColor,
                    scale_factor=0.013, texture=None, colli_obj=None, material=None,
                    **kwargs):
        '''
        load object to the physical world
        '''
        args = locals()
        del args['self']
        # print(args)
        shift1 = [0, 0.1, 0]
        # shift2 = [0, 0, 0]
        object_name = random.choice(objectname) 
        if ele == "Table":
            basePosition = self._obj_args[ele]["basePosition"]
            baseOrientation = self._obj_args[ele]["baseOrientation"]
        elif ele != "Coffee Cup BB":
            basePosition, baseOrientation = transform(ele, self._configs[ele], object_name)
        else:
            basePosition, baseOrientation = transform('Coffee Plate BB', self._configs['Coffee Plate BB'], object_name)
        if 'Napkin Cloth BB' == ele:
            basePosition[2] += 0.06
        elif 'Dinner Plate BB' == ele:
            basePosition[2] += 0.04
        elif 'Coffee Cup BB' == ele:
            basePosition[2] += 0.04
            baseOrientation = pb.getQuaternionFromEuler([math.pi/2., 0, -math.pi/6.])
        elif "Table Cloth Sides" == ele:
            # basePosition[2] -= 0.02
            return None, None, None
        else:
            basePosition[2] += 0.02
        self._obj_args[ele]['basePosition'] = basePosition
        self._obj_args[ele]['basePosition'][2] += 0.02
        self._obj_args[ele]['baseOrientation'] = baseOrientation
        pos = basePosition
        rot = baseOrientation
        scale = scale_factor
        mesh = nvisii.mesh.get(object_name)
        mesh_type = self.mesh_type[object_name] 
        if mesh_type in ["fork", "knife", "spoon"]:
            basePosition[2] += 0.02
            pos[2] += 0.02
        vertices = mesh.get_vertices()
        hsv = nvisii.texture.get(f"hsv_{object_name}") 
        tex = nvisii.texture.get(object_name) 
        if hsv is None and tex is not None:
            hsv = nvisii.texture.create_hsv(f"hsv_{object_name}", tex, 
                hue = 0, saturation = 1.0, value = 1.0, mix = 1.0)
        # visualShapeId = self.client.createVisualShape(
        #             pb.GEOM_MESH,
        #             vertices = vertices,
        #             meshScale=[scale_factor, scale_factor, scale_factor])
        collisionShapeId = self.client.createCollisionShape(
                    pb.GEOM_MESH,
                    vertices = vertices,
                    meshScale=[scale_factor, scale_factor, scale_factor])
        # if fileName is not None:
        #     if colli_obj is not None:
        #         collisionShapeId = self.client.createCollisionShape(
        #             shapeType=shapeType,
        #             fileName=colli_obj,
        #             meshScale=[scale_factor, scale_factor, scale_factor])
        #     else:
        #         if shapeType == 3:
        #             collisionShapeId = self.client.createCollisionShape(
        #             shapeType=pb.GEOM_BOX,
        #             halfExtents=kwargs['kwargs']['halfExtents'],
        #             collisionFramePosition=shift1,
        #             )
        #         else:
        #             collisionShapeId = self.client.createCollisionShape(
        #                 shapeType=5,
        #                 fileName=fileName,
        #                 meshScale=[scale_factor, scale_factor, scale_factor]
        #             )
        # else:
        #     collisionShapeId = self.client.createCollisionShape(
        #         shapeType=shapeType,
        #         meshScale=[scale_factor, scale_factor, scale_factor],
        #         **kwargs)

        multiBodyId = self.client.createMultiBody(
            baseMass=baseMass,
            baseCollisionShapeIndex=collisionShapeId, 
            # baseVisualShapeIndex=visualShapeId,
            basePosition=basePosition,
            baseOrientation=baseOrientation) 
        name = f"object_{multiBodyId}" 
        object_nvisii = nvisii.entity.get(name)
        if object_nvisii is None:
            object_nvisii = nvisii.entity.create(
                name=name,
                mesh=mesh,
                transform = nvisii.transform.get(name) \
                    if nvisii.transform.get(name) is not None \
                    else nvisii.transform.create(name),
                material = nvisii.material.get(name) \
                    if nvisii.material.get(name) is not None \
                    else nvisii.material.create(name),
            )
        else:
            object_nvisii.clear_mesh()
            object_nvisii.set_mesh(mesh)
        

        mat = nvisii.material.get(name)

        # sample whether load the texture
        
        if object_name not in ["table", "table_cloth", "napkin_cloth"]:
            random_num = random.uniform(0, 1)
        else:
            random_num = 0
        if tex is not None and random_num < 0.3: 
            # if load the texture, then record the name from predefined file
            mat.clear_base_color_texture()
            mat.clear_metallic_texture()
            mat.clear_transmission_texture()
            mat.clear_roughness_texture()
            mat.clear_sheen_texture()
            mat.clear_ior_texture()
            # mat.clear_base_color()
            mat.set_base_color_texture(hsv)
            if material == 'wood':
                mat.set_transmission(0.0)  # should 0 or 1      
                mat.set_roughness(0.75) # default is 1 set_clearcoat_roughness(clearcoatRoughness) 
                mat.set_metallic(0)
                mat.set_roughness_texture(tex)
            elif material == 'cloth':
                mat.set_transmission(0.0)  # should 0 or 1      
                mat.set_roughness(1.0) # default is 1 set_clearcoat_roughness(clearcoatRoughness) 
                mat.set_sheen_texture(tex)
            else:
                mat.set_roughness(.5)
                mat.set_metallic_texture(tex)
            self.object_property[multiBodyId - 3] = self.texture_name[object_name]
        else:
            # sample a material
            material_choice = random.choices(
                list(material_weights[mesh_type].keys()),
                weights=material_weights[mesh_type].values(),
                k=1
            )[0]

            # This is a simple logic for more natural random materials, e.g.,  
            # mirror or glass like objects
            if material_choice == 'plastic' :
                mat.set_transmission(0)  # should 0 or 1  
                if mesh_type in ["plate", "square plate"] :
                    mat.set_metallic(0.1)  # should 0 or 1      
                    mat.set_roughness(0.1) # default is 1  
                else:
                    mat.set_metallic(0)  # should 0 or 1      
                    mat.set_roughness(0.5) # default is 1  
            elif material_choice == 'metallic':
                mat.set_metallic(1)  # should 0 or 1      
                mat.set_transmission(0)  # should 0 or 1      
                mat.set_roughness(0) # default is 1 
            # elif material_choice == 'glass' :
            #     mat.set_metallic(0.1)  # should 0 or 1   
            #     mat.set_transmission(1.0)  # should 0 or 1      
            #     mat.set_roughness(0)
            #     mat.set_ior(1.7)
                # mat.set_subsurface_radius(nvisii.vec3(0.0, .0, .0))
            
            colors_dict = color_weights[mesh_type]
            color_selected = random.choices(
                    list(colors_dict.keys()), weights=colors_dict.values(), k=1
                )[0]
            if mesh_type in ["plate", "square plate"] and material_choice == 'glass':
                color_selected = 'pure_white'
            rgb = colors[random.choice(
                self.color_list[color_selected]
            )]            
            # print(rgb)
            mat.clear_base_color_texture()
            mat.clear_metallic_texture()
            mat.clear_transmission_texture()
            mat.clear_ior_texture()
            mat.set_base_color([rgb[0]/255, rgb[1]/255, rgb[2]/255])
            if mesh_type in ["spoon", "knife", "fork", "wine glass"]:
                self.object_property[multiBodyId- 3] = mesh_type
            
            elif material_choice == 'glass':
                if mesh_type in ['plate', 'square plate', 'cup'] or color_selected == 'white':
                    self.object_property[multiBodyId - 3] = material_choice+ " "\
                        + mesh_type
                else:
                    self.object_property[multiBodyId - 3] = color_selected + " "\
                        + material_choice+ " " + mesh_type
            else:
                # r = random.randint(0, 1)
                # if r == 0:
                self.object_property[multiBodyId - 3] = color_selected + " "\
                        + mesh_type
                # else:
                #     self.object_property[multiBodyId - 3] = mesh_type
 
        object_nvisii.get_transform().set_position(pos)
        object_quat = nvisii.normalize(nvisii.quat(rot[0],rot[1],rot[2],rot[3]))
        object_nvisii.get_transform().set_rotation(object_quat)
        object_nvisii.get_transform().set_scale((scale, scale, scale))
        id = object_nvisii.get_id()
        self.id_transfer[id] = multiBodyId - 3
        self.id_transfer_2[multiBodyId - 3] = id
        self.ids_pybullet_and_nvisii_names.append(
        {
            "pybullet_id": multiBodyId, 
            "nvisii_id": name,
        })
        return multiBodyId, args, object_name 

    def load_object(self, type_name, mesh_name, baseMass, position, angle, rgb, size,
                    name = None, scale_factor=0.013, material=None, texture=False):
        '''
        load object to the physical world
        '''
        
        # self.grid[position[0]:position[0]+size[1], 
        #         position[1]: position[1]+size[0]] = 1
        if self.indivisual_loading:
            for ele in self.object_conf:
                if self.object_conf[ele] is not None:
                    if mesh_name in self.object_conf[ele]:
                        obj = mesh_name
                        mesh_path = self.object_conf[ele][obj]["meshes"]
                        texture_path = self.object_conf[ele][obj]["texture"]
                        self.texture_name[obj] = self.object_conf[ele][obj]["name"]
                        print("loading {}".format(self.object_conf[ele][obj]["name"]))
                        if obj in self.mesh_type.keys():
                            pass
                        else:
                            self.mesh_type[obj] = ele 
                            nvisii.mesh.create_from_file(obj, mesh_path)
                            if texture_path is not None:
                                nvisii.texture.create_from_file(obj, texture_path)
        if name is not None:
            self.obj_position.update({
                name: position
            })
        my_unique_name = name
        _, initorn = transform(type_name, self._configs[type_name], mesh_name)
        baseOrientationQuat = pb.getQuaternionFromEuler([0, 0, angle/180 * np.pi])
        matrix_orn = np.matrix(self.client.getMatrixFromQuaternion(initorn)).reshape(3,3)
        matrix_orn_2 = np.matrix(self.client.getMatrixFromQuaternion(baseOrientationQuat)).reshape(3,3)
        baseOrientation = getQuaternionFromMatrix(matrix_orn_2 * matrix_orn)
        xyz = [(position[1] + size[0])/40 -0.4, 
                (position[0] + size[1])/40 - 0.4]
        if len(position)== 3:
            basePosition = [xyz[0], xyz[1], 1.15 + position[2]*0.1]
        else: 
            basePosition = [xyz[0], xyz[1], 1.15]
        pos = basePosition
        rot = baseOrientation
        scale = scale_factor
        mesh = nvisii.mesh.get(mesh_name)
        mesh_type = self.mesh_type[mesh_name] 
        if mesh_type in ["fork", "knife", "spoon"]:
            basePosition[2] += 0.02
            pos[2] += 0.02
        vertices = mesh.get_vertices()
        hsv = nvisii.texture.get(f"hsv_{mesh_name}") 
        tex = nvisii.texture.get(mesh_name) 
        if hsv is None and tex is not None:
            hsv = nvisii.texture.create_hsv(f"hsv_{mesh_name}", tex, 
                    hue = 0, saturation = 1.0, value = 1.0, mix = 1.0)

        collisionShapeId = self.client.createCollisionShape(
                    pb.GEOM_MESH,
                    vertices = vertices,
                    meshScale=[scale_factor, scale_factor, scale_factor])

        multiBodyId = self.client.createMultiBody(
                    baseMass=baseMass,
                    baseCollisionShapeIndex=collisionShapeId, 
                    basePosition=basePosition,
                    baseOrientation=baseOrientation) 
        
        name = f"object_{multiBodyId}" 
        object_nvisii = nvisii.entity.get(name)
        if object_nvisii is None:
            object_nvisii = nvisii.entity.create(
                name=name,
                mesh=mesh,
                transform = nvisii.transform.get(name) \
                    if nvisii.transform.get(name) is not None \
                    else nvisii.transform.create(name),
                material = nvisii.material.get(name) \
                    if nvisii.material.get(name) is not None \
                    else nvisii.material.create(name),
            )
        else:
            object_nvisii.clear_mesh()
            object_nvisii.set_mesh(mesh)
        
        mat = nvisii.material.get(name)

        # if mesh_name not in ["table", "table_cloth", "napkin_cloth"]:
        #     random_num = random.uniform(0, 1)
        # else:
        #     random_num = 0
        if tex is not None and texture: 
            # if load the texture, then record the name from predefined file
            mat.clear_base_color_texture()
            mat.clear_metallic_texture()
            mat.clear_transmission_texture()
            mat.clear_roughness_texture()
            mat.clear_sheen_texture()
            mat.clear_ior_texture()
            mat.set_base_color_texture(hsv)
            if material == 'wood':
                mat.set_transmission(0.0)  # should 0 or 1      
                mat.set_roughness(0.75) # default is 1 set_clearcoat_roughness(clearcoatRoughness) 
                mat.set_metallic(0)
                mat.set_roughness_texture(tex)
            elif material == 'cloth':
                mat.set_transmission(0.0)  # should 0 or 1      
                mat.set_roughness(1.0) # default is 1 set_clearcoat_roughness(clearcoatRoughness) 
                mat.set_sheen_texture(tex)
            else:
                mat.set_transmission(0.0)  # should 0 or 1      
                mat.set_roughness(1.0)
                mat.set_sheen_texture(tex)
        else:

            # This is a simple logic for more natural random materials, e.g.,  
            # mirror or glass like objects
            if material == 'plastic' :
                mat.set_transmission(0)  # should 0 or 1  
                if mesh_type in ["plate", "square plate"] :
                    mat.set_metallic(0.1)  # should 0 or 1      
                    mat.set_roughness(0.1) # default is 1  
                else:
                    mat.set_metallic(0)  # should 0 or 1      
                    mat.set_roughness(0.5) # default is 1  
            elif material == 'metallic':
                mat.set_metallic(1)  # should 0 or 1      
                mat.set_transmission(0)  # should 0 or 1      
                mat.set_roughness(0) # default is 1 
            elif material == 'cloth':
                mat.set_transmission(0.0)  # should 0 or 1      
                mat.set_roughness(1.0) # default is 1 set_clearcoat_roughness(clearcoatRoughness) 
                mat.set_sheen_texture(tex)
            
            mat.clear_base_color_texture()
            mat.clear_metallic_texture()
            mat.clear_transmission_texture()
            mat.clear_ior_texture()
            mat.set_base_color([rgb[0]/255, rgb[1]/255, rgb[2]/255])

        object_nvisii.get_transform().set_position(pos)
        object_quat = nvisii.normalize(nvisii.quat(rot[0],rot[1],rot[2],rot[3]))
        object_nvisii.get_transform().set_rotation(object_quat)
        object_nvisii.get_transform().set_scale((scale, scale, scale))
        nvisii_id = object_nvisii.get_id()
        self.ids_pybullet_and_nvisii_names.append(
        {
            "pybullet_id": multiBodyId, 
            "nvisii_id": name,
        })
        if my_unique_name is not None:
            self.name_id_dict[my_unique_name]= {
                    "pybullet_id": multiBodyId, 
                    "nvisii_id": name,
                }
        # if np.any(self.grid[position[0]:position[0]+size[1], 
        #         position[1]: position[1]+size[0]]):
        #     return False
        # else:
        # return True

    def _load_world_from_config(self):
        '''
        load all the objects to the physical engine according to the json config file
        '''
         
        for ele in self._obj_args:
            cate = self._obj_args[ele].pop('category')
            # if ele == 'Table Cloth Sides':
            #     self._obj_args[ele]['basePosition'][2]  -= 0.02
            # # elif 'Wine Glass' in ele:
            # #     self._obj_args[ele]['basePosition'][2] = 1.7
            # else:
            #     self._obj_args[ele]['basePosition'][2] += 0.02
            obj_id, _, mesh_name = self._load_object(ele,
                **self._obj_args[ele]
            )
            if obj_id is None:
                continue
            self.obj_ids[cate].append(obj_id - 3)
            self.obj_names[ele] = obj_id - 3
            if ele in self.mentioned_objects:
                for ref in self.bb2ref[ele]:
                    if ref in self.instruction:
                        self.instruction = self.instruction.replace(ref, "a " + self.object_property[obj_id-3]) 
                        break
            self.obj_mesh_name[ele] = mesh_name
        # print(self.obj_names)
    
    def _load_world_from_bullet(self):
        print(self.filepath)
        self.client.restoreState(fileName=self.filepath + 'scene.bullet')

    def _load_instruction(self):
        json_path = get_json_path_from_scene(self.mode, self._state_id, self._step_id, self._table_id)
        if os.path.isfile(json_path):
            data = read_json(json_path)
            self.instruction = data['instruction']
            self.instruction = self.instruction.lower()

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

    def get_observation_nvisii_cliport(self, filename):
        depth_pkl = ()
        intrinsics_list = []
        for ids in self.ids_pybullet_and_nvisii_names:

            # get the pose of the objects
            pos, rot = self.client.getBasePositionAndOrientation(ids['pybullet_id'])

            # get the nvisii entity for that object
            obj_entity = nvisii.entity.get(ids['nvisii_id'])
            obj_entity.get_transform().set_position(pos)

            # nvisii quat expects w as the first argument
            obj_entity.get_transform().set_rotation(rot)
        for index in range(3):
            nvisii.set_camera_entity(self.camera_nvisii_list[index + 1])
            nvisii.render_to_file(
                width=640, 
                height=480, 
                samples_per_pixel=64,
                file_path=filename + f'rgb{index}.png'
            )
            dep = nvisii.render_data(
                width=640,
                height=480, 
                start_frame=0,
                frame_count=10,
                bounce=0,
                options="depth",
                # file_path=filename + f"/depth{index}.png"
            )
            # print(dep[0],dep[1],dep[2],dep[3])
            depth_image_size = (480, 640, 4)
            depth_array = np.array(dep).reshape(depth_image_size)
            # depth_array = np.array(depth_array).reshape(opt.height,opt.width,4)
            depth_array = np.flipud(depth_array)
            # depth = (10 - zbuffer * (10 - 0.01))
            # depth = (10 * 0.01) / depth
            depth_pkl += (depth_array[:, :, 0],)
        with open(filename + 'depth.pkl', 'wb') as f:
            pickle.dump(depth_pkl, f) 

    @staticmethod 
    def convert_from_uvd(u, v, d,fx,fy,cx,cy):
        # d *= self.pxToMetre
        x_over_z = (cx - u) / fx
        y_over_z = (cy - v) / fy
        z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
        x = x_over_z * z
        y = y_over_z * z
        return x, y, z
    
    def get_observation_nvisii_image(self, filename=None):
        for ids in self.ids_pybullet_and_nvisii_names:

            # get the pose of the objects
            pos, rot = self.client.getBasePositionAndOrientation(ids['pybullet_id'])

            # get the nvisii entity for that object
            obj_entity = nvisii.entity.get(ids['nvisii_id'])
            obj_entity.get_transform().set_position(pos)

            # nvisii quat expects w as the first argument
            obj_entity.get_transform().set_rotation(rot)
        # self.update_visual_objects([self.robot_id, self.ee.body, self.ee.base], '.', self.nv_objects)
        nvisii.set_camera_entity(self.camera_nvisii_list[0])
        if filename is None:
            filename = 'video/'
        nvisii.render_to_file(
            width=480, 
            height=480, 
            samples_per_pixel=64,
            file_path=filename + f'rgb_{self.video_frame_counter}.png'
        ) 
        nvisii.set_camera_entity(self.camera_nvisii_list['gui'])
        nvisii.render_to_file(
            width=640, 
            height=480, 
            samples_per_pixel=64,
            file_path=filename + f'rgb_gui_{self.video_frame_counter}.png'
        ) 
        self.video_frame_counter += 1
        # img = Image.open(filename + 'rgb.png').convert('RGB')
        # return np.array(img, dtype=np.uint8)



    def get_observation_nvisii(self, filename):
        for ids in self.ids_pybullet_and_nvisii_names:

            # get the pose of the objects
            pos, rot = self.client.getBasePositionAndOrientation(ids['pybullet_id'])

            # get the nvisii entity for that object
            obj_entity = nvisii.entity.get(ids['nvisii_id'])
            obj_entity.get_transform().set_position(pos)

            # nvisii quat expects w as the first argument
            obj_entity.get_transform().set_rotation(rot)
        nvisii.set_camera_entity(self.camera_nvisii_list[0])
        
        nvisii.render_to_file(
            width=640, 
            height=640, 
            samples_per_pixel=64,
            file_path=filename
        ) 
        mask = nvisii.render_data(
            width=640,
            height=640, 
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
            # file_path=filename+ "/mask.png"
        )
        mask = np.array(mask).reshape(640, 640, 4)[:, :, 0]
        mask = np.flipud(mask)
        mask = np.uint8(mask)
        # print('min',np.min(mask))
        # print('max',np.max(mask))
        return mask



    def get_obj_pose(self, id):
        return self.client.getBasePositionAndOrientation(id)

    def reset_obj_pose(self, name, position, size, baseOrientationAngle):
        # print(self.name_id_dict)
        id, nvisii_id = (
            self.name_id_dict[name]["pybullet_id"],
            self.name_id_dict[name]["nvisii_id"])
        # if np.any(self.grid[position[0]:position[0]+size[1], 
        #     position[1]: position[1]+size[0]]):
        #     return False
        # self.grid[position[0]:position[0]+size[1], 
        #     position[1]: position[1]+size[0]] = 1
        pre_position = self.obj_position[name]
        xyz = [(position[1] + size[0])/40 -0.4, 
                (position[0] + size[1])/40 - 0.4]
        if len(position) == 3:
            basePosition = [xyz[0], xyz[1], 1.15 + position[2]*0.1]
        else:
            basePosition = [xyz[0], xyz[1], 1.15]

        self.grid[pre_position[0]:pre_position[0]+size[1],
            pre_position[1]: pre_position[1]+size[0]] = 0
        baseOrientationQuat = pb.getQuaternionFromEuler([0, 0, baseOrientationAngle/180 * np.pi])
        _, orn = self.client.getBasePositionAndOrientation(id)
        matrix_orn = np.matrix(self.client.getMatrixFromQuaternion(orn)).reshape(3,3)
        matrix_orn_2 = np.matrix(self.client.getMatrixFromQuaternion(baseOrientationQuat)).reshape(3,3)
        new_orn = getQuaternionFromMatrix(matrix_orn_2 * matrix_orn)
        self.client.resetBasePositionAndOrientation(
            bodyUniqueId=id, 
            posObj=basePosition, 
            ornObj=new_orn)
        # get the nvisii entity for that object
        obj_entity = nvisii.entity.get(nvisii_id)
        obj_entity.get_transform().set_position(basePosition)

        # nvisii quat expects w as the first argument
        obj_entity.get_transform().set_rotation(new_orn)

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
    
    def apply_action(self, action, interp_n=INTERPOLATE_NUM):
        timeout = self.primitive(self._movej, self._movep, self.ee, action, interp_n)
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

    def _movej(self, targj, speed=0.01, timeout=5, interp_n=INTERPOLATE_NUM):
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
        res, path = self.planner.plan_start_goal(start=currj, goal=targj, interp_n=interp_n)
        if self._use_gui:
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_RENDERING, 1)
        if res:
            self.planner.execute(path, dynamics=False, video=self.record_video, 
                get_obs=self.get_observation_nvisii_image, gui_writer=self.gui_writer, get_gui_obs=self.get_render_img)
            return False
        else:
            print(f'Warning: _movej exceeded {timeout} second timeout. Skipping.')
        return True
        # pass

    def _movep(self, pose, speed=0.01, interp_n=INTERPOLATE_NUM):
        """Move UR5 to target end effector pose."""
        pos, orn = pose
        # orn = [0.,0.,0.,1.]
        pose = (pos, orn)
        targj = self._solve_ik(pose)
        # print("goal: {}".format(targj))
        return self._movej(targj, speed, interp_n=interp_n)

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

    def __del__(self):
        nvisii.deinitialize()
        # del self.client

    ###########################################
    # PDDLStream used functions
    ###########################################

