import os
from typing import Tuple 
import nvisii
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import numpy as np
from tabletop_gym.envs.utils import camera_coord, FOV, CAMERA_MATRIX
from PIL import Image
from tabletop_gym.envs.utils import read_json, OBJ_CONFIG_10_B, colors_name_train, \
    color_weights, OBJECT_OBJ_MAT_CONF, material_weights, colors, transform

def load_materials():
    object_conf = read_json(OBJECT_OBJ_MAT_CONF)
    texture_name = {}
    mesh_type = {}
    for ele in object_conf:
        if object_conf[ele] is not None:
            for obj in object_conf[ele]:
                mesh_path = object_conf[ele][obj]["meshes"]
                texture_path = object_conf[ele][obj]["texture"]
                texture_name[obj] = object_conf[ele][obj]["name"]
                mesh_type[obj] = ele
                nvisii.mesh.create_from_file(obj, mesh_path)
                if texture_path is not None:
                    nvisii.texture.create_from_file(obj, texture_path)
    return texture_name, mesh_type

def create_object(
    name: str, 
    meshfile: str,
    pos: Tuple,
    rot: Tuple,
    scale: Tuple,
    rgb: Tuple,
    material: str,
    mesh_type: str,
    texfile: str = None,
):
    mesh = nvisii.mesh.create_from_file(name, meshfile)
    if texfile is not None:
        tex = nvisii.texture.create_from_file(f"{name}_tex", texfile)
        hsv = nvisii.texture.create_hsv(name, tex, 
            hue = 0, saturation = 0.6, value = 5.0, mix = 1.0)
    
    object_nvisii = nvisii.entity.create(
        name=name,
        mesh = mesh,
        transform = nvisii.transform.create(name),
        material = nvisii.material.create(name)
    )
    object_nvisii.get_transform().set_position(pos)
    object_quat = nvisii.normalize(nvisii.quat(rot[0],rot[1],rot[2],rot[3]))
    object_nvisii.get_transform().set_rotation(object_quat)
    object_nvisii.get_transform().set_scale(scale)

    mat = nvisii.material.get(name)
    mat.set_roughness(.5)
    if texfile is not None:
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
    else:
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
        mat.set_base_color([rgb[0]/255, rgb[1]/255, rgb[2]/255])
    # Set the collision with the floor mesh
    # first lets get the vertices 
    vertices = object_nvisii.get_mesh().get_vertices()

    # get the position of the object
    pos = object_nvisii.get_transform().get_position()
    pos = [pos[0],pos[1],pos[2]]
    scale = object_nvisii.get_transform().get_scale()
    scale = [scale[0],scale[1],scale[2]]
    rot = object_nvisii.get_transform().get_rotation()
    rot = [rot[0],rot[1],rot[2],rot[3]]

    # create a collision shape that is a convex hull
    obj_col_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices = vertices,
        # fileName ="./tabletop_gym/envs/assets/objects/table.obj",
        meshScale = scale,
    )

    # create a body without mass so it is static
    p.createMultiBody(
        baseCollisionShapeIndex = obj_col_id,
        basePosition = pos,
        baseOrientation= rot,
    )
    return object_nvisii, obj_col_id 


nvisii.initialize(headless=True, lazy_updates=True)
nvisii.enable_denoiser()
counter = 0
# texture_name, mesh_type = load_materials()
objects_lists = read_json(OBJ_CONFIG_10_B)
objects_config = read_json(OBJECT_OBJ_MAT_CONF)
for ele in objects_lists:
    obj_class = objects_lists[ele]['objectclass']
    for obj_name in objects_lists[ele]['objectname']:
        if obj_name not in objects_config[obj_class]:
            continue
        name = objects_config[obj_class][obj_name]['name']
        mesh = objects_config[obj_class][obj_name]['meshes']
        texture = objects_config[obj_class][obj_name]['texture']
        if texture is not None:
            nvisii.clear_all()
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
                at = (0.0, 0.0, 1.07),
                up = (0,0,1),
                eye = (0, 0, 1.7),
            )
            seconds_per_step = 1.0 / 240.0
            frames_per_second = 30.0
            physicsClient = p.connect(p.DIRECT) # non-graphical version
            p.setGravity(0,0,-10)
            sun = nvisii.entity.create(
                name = "sun",
                mesh = nvisii.mesh.create_sphere("sphere"),
                transform = nvisii.transform.create("sun"),
                light = nvisii.light.create("sun")
            )
            sun.get_transform().set_position((10,10,10))
            sun.get_light().set_temperature(4780)
            sun.get_light().set_intensity(500)

            nvisii.set_dome_light_intensity(1.0)

            # atmospheric thickness makes the sky go orange, almost like a sunset
            nvisii.set_dome_light_sky(sun_position=(10,10,10), atmosphere_thickness=1.0, saturation=1.0)

            # nvisii_id, obj_col_id = create_object(
            #     name ='table', 
            #     meshfile='./tabletop_gym/envs/assets/objects/table.obj',
            #     # texfile='./tabletop_gym/envs/assets/objects/Table_BaseColor.tga',
            #     pos = (0.0,0.0,0.0),
            #     rot = (0.5, 0.5, 0.5, 0.5),
            #     scale = (0.013, 0.013, 0.013)
            # )
            if obj_class in ['fork', 'knife', 'spoon']:
                rot =(0.5, 0.5, 0.5, 0.5) 
                scale = (1.0, 1.0, 1.0)
            elif obj_class in ['plate']: 
                rot = [
                    -2.0912712868518937e-05,
                    0.00045780070514357554,
                    0.071072590118122,
                    0.7071061548551073
                ]
                scale = (0.9, 0.9, 0.9)
                pos = (0.0,0.0,1.09)
            elif "mug" in obj_name:
                scale = (1.0, 1.0, 1.0)
                rot = [
                    0.0004946447825311935,
                    -0.0003438817232483529,
                    0.7071095092755839,
                    0.7071037964570283
                    ]
                pos = (0.0,-0.05,1.09)
            elif 'wine' in obj_name:
                scale = (1.0, 1.0, 1.0)
                rot = [
                        0.0004946447825311935,
                        -0.0003438817232483529,
                        0.071095092755839,
                        0.7071037964570283
                        ]
                pos = (0.0,0.12,1.09) 
            else:
                scale = (1.0, 1.0, 1.0)
                rot = [
                    0.0004946447825311935,
                    -0.0003438817232483529,
                    0.071095092755839,
                    0.7071037964570283
                    ]
                pos = (0.0,0.0,1.09)
            nvisii_id, obj_col_id = create_object(
                name ='fork', 
                meshfile=mesh,
                # texfile='./tabletop_gym/envs/assets/objects/Table_BaseColor.tga',
                rot = rot,
                pos = pos,
                scale = scale,
                rgb=None,
                material = mat,
                mesh_type=obj_class,
                texfile=texture
            )

            opt = lambda : None
            opt.nb_objects = 10
            opt.spp = 64 
            opt.width = 200
            opt.height = 200
            opt.noise = False
            opt.frame_freq = 8
            opt.nb_frames = 100
            opt.outf = '03_pybullet'
            nvisii.set_camera_entity(camera)
            nvisii.render_to_file(
                width=int(opt.width), 
                height=int(opt.height), 
                samples_per_pixel=int(opt.spp),
                file_path=f"{opt.outf}/{obj_name}-{color}-{counter}.png"
            )
            counter += 1
        if obj_class not in color_weights:
            continue
        for color in color_weights[obj_class]:
            if color_weights[obj_class][color] > 0:
                for rgb_name in colors_name_train[color]:
                    rgb = colors[rgb_name]
                    for mat in material_weights[obj_class]:
                        if material_weights[obj_class][mat] > 0:

                            nvisii.clear_all()
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
                                at = (0.0, 0.0, 1.07),
                                up = (0,0,1),
                                eye = (0, 0, 1.7),
                            )
                            seconds_per_step = 1.0 / 240.0
                            frames_per_second = 30.0
                            physicsClient = p.connect(p.DIRECT) # non-graphical version
                            p.setGravity(0,0,-10)
                            sun = nvisii.entity.create(
                                name = "sun",
                                mesh = nvisii.mesh.create_sphere("sphere"),
                                transform = nvisii.transform.create("sun"),
                                light = nvisii.light.create("sun")
                            )
                            sun.get_transform().set_position((10,10,10))
                            sun.get_light().set_temperature(5780)
                            sun.get_light().set_intensity(1000)

                            nvisii.set_dome_light_intensity(1.0)

                            # atmospheric thickness makes the sky go orange, almost like a sunset
                            nvisii.set_dome_light_sky(sun_position=(10,10,10), atmosphere_thickness=1.0, saturation=1.0)

                            # nvisii_id, obj_col_id = create_object(
                            #     name ='table', 
                            #     meshfile='./tabletop_gym/envs/assets/objects/table.obj',
                            #     # texfile='./tabletop_gym/envs/assets/objects/Table_BaseColor.tga',
                            #     pos = (0.0,0.0,0.0),
                            #     rot = (0.5, 0.5, 0.5, 0.5),
                            #     scale = (0.013, 0.013, 0.013)
                            # )
                            if obj_class in ['fork', 'knife', 'spoon']:
                                rot =(0.5, 0.5, 0.5, 0.5) 
                                scale = (1.0, 1.0, 1.0)
                            elif obj_class in ['plate']: 
                                rot = [
                                    -2.0912712868518937e-05,
                                    0.00045780070514357554,
                                    0.071072590118122,
                                    0.7071061548551073
                                ]
                                scale = (0.9, 0.9, 0.9)
                                pos = (0.0,0.0,1.09)
                            elif 'wine' in obj_name:
                                scale = (1.0, 1.0, 1.0)
                                rot = [
                                    0.0004946447825311935,
                                    -0.0003438817232483529,
                                    0.071095092755839,
                                    0.7071037964570283
                                    ]
                                pos = (0.0,0.12,1.09) 
                            elif "mug" in obj_name:
                                scale = (1.0, 1.0, 1.0)
                                rot = [
                                    0.0004946447825311935,
                                    -0.0003438817232483529,
                                    0.7071095092755839,
                                    0.7071037964570283
                                    ]
                                pos = (0.0,-0.05,1.09)
                            else:
                                scale = (1.0, 1.0, 1.0)
                                rot = [
                                    0.0004946447825311935,
                                    -0.0003438817232483529,
                                    0.071095092755839,
                                    0.7071037964570283
                                    ]
                                pos = (0.0,0.0,1.09)
                            nvisii_id, obj_col_id = create_object(
                                name ='fork', 
                                meshfile=mesh,
                                # texfile='./tabletop_gym/envs/assets/objects/Table_BaseColor.tga',
                                pos = pos,
                                rot = rot,
                                scale = scale,
                                rgb=rgb,
                                material = mat,
                                mesh_type=obj_class
                            )

                            opt = lambda : None
                            opt.nb_objects = 10
                            opt.spp = 64 
                            opt.width = 200
                            opt.height = 200
                            opt.noise = False
                            opt.frame_freq = 8
                            opt.nb_frames = 100
                            opt.outf = '03_pybullet'
                            nvisii.set_camera_entity(camera)
                            nvisii.render_to_file(
                                width=int(opt.width), 
                                height=int(opt.height), 
                                samples_per_pixel=int(opt.spp),
                                file_path=f"{opt.outf}/{ele}-{obj_name}-{color}-{counter}.png"
                            )
                            counter += 1
p.disconnect()
nvisii.deinitialize()