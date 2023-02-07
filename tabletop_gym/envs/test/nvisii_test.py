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

opt = lambda : None
opt.nb_objects = 10
opt.spp = 64 
opt.width = 640
opt.height = 640 
opt.noise = False
opt.frame_freq = 8
opt.nb_frames = 100
opt.outf = '03_pybullet'

def srgb_to_linsrgb (srgb):
    """Convert sRGB values to physically linear ones. The transformation is
       uniform in RGB, so *srgb* can be of any shape.

       *srgb* values should range between 0 and 1, inclusively.

    """
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return np.where (srgb > 0.04045, gamma, scale)


def create_object(
    name: str, 
    meshfile: str,
    texfile: str,
    pos: Tuple,
    rot: Tuple,
    scale: Tuple,

):
    mesh = nvisii.mesh.create_from_file(name, meshfile)
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
    mat.set_base_color_texture(hsv)
    mat.set_roughness_texture(tex)

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




# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

# show an interactive window, and use "lazy" updates for faster object creation time 
nvisii.initialize(headless=True, lazy_updates=True)

if not opt.noise is True: 
    nvisii.enable_denoiser()

# Create a camera
camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create_from_fov(
        name = "camera", 
        field_of_view = FOV/180 * np.pi,
        aspect = 1.0
    )
)
print(camera_coord())
camera.get_transform().look_at(
    at = (0.08, 0.02, 1.07),
    up = (0,0,1),
    eye = camera_coord(),
)
nvisii.set_camera_entity(camera)

# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0
physicsClient = p.connect(p.DIRECT) # non-graphical version
p.setGravity(0,0,-10)

# Lets set the scene

# Change the dome light intensity
nvisii.set_dome_light_intensity(1.0)

# atmospheric thickness makes the sky go orange, almost like a sunset
nvisii.set_dome_light_sky(sun_position=(10,10,10), atmosphere_thickness=1.0, saturation=1.0)

# Lets add a sun light
sun = nvisii.entity.create(
    name = "sun",
    mesh = nvisii.mesh.create_sphere("sphere"),
    transform = nvisii.transform.create("sun"),
    light = nvisii.light.create("sun")
)
sun.get_transform().set_position((10,10,10))
sun.get_light().set_temperature(5780)
sun.get_light().set_intensity(1000)

# floor = nvisii.entity.create(
#     name="floor",
#     mesh = nvisii.mesh.create_plane("floor"),
#     transform = nvisii.transform.create("floor"),
#     material = nvisii.material.create("floor")
# )
# floor.get_transform().set_position((0,0,0))
# floor.get_transform().set_scale((10, 10, 10))
# floor.get_material().set_roughness(0.1)
# floor.get_material().set_base_color((0.5,0.5,0.5))

# # Set the collision with the floor mesh
# # first lets get the vertices 
# vertices = floor.get_mesh().get_vertices()

# # get the position of the object
# pos = floor.get_transform().get_position()
# pos = [pos[0],pos[1],pos[2]]
# scale = floor.get_transform().get_scale()
# scale = [scale[0],scale[1],scale[2]]
# rot = floor.get_transform().get_rotation()
# rot = [rot[0],rot[1],rot[2],rot[3]]

# # create a collision shape that is a convex hull
# obj_col_id = p.createCollisionShape(
#     p.GEOM_MESH,
#     vertices = vertices,
#     meshScale = scale,
# )

# # create a body without mass so it is static
# p.createMultiBody(
#     baseCollisionShapeIndex = obj_col_id,
#     basePosition = pos,
#     baseOrientation= rot,
# )    

nvisii_id, obj_col_id = create_object(
    name ='table', 
    meshfile='./tabletop_gym/envs/assets/objects/table.obj',
    texfile='./tabletop_gym/envs/assets/objects/Table_BaseColor.tga',
    pos = (0.0,0.0,0.0),
    rot = (0.5, 0.5, 0.5, 0.5),
    scale = (0.013, 0.013, 0.013)
)

# mesh = nvisii.mesh.create_from_file(
#     "table", 
#     "./tabletop_gym/envs/assets/objects/table.obj"
# )
# table_tex = nvisii.texture.create_from_file(
#     "table_tex",
#     './tabletop_gym/envs/assets/objects/Table_BaseColor.tga'
# )

# Textures can be mixed and altered. 
# Checkout create_hsv, create_add, create_multiply, and create_mix
# table_hsv = nvisii.texture.create_hsv("table", table_tex, 
#     hue = 0, saturation = 0.6, value = 5.0, mix = 1.0)
# # create a table
# table = nvisii.entity.create(
#     name="table",
#     mesh = mesh,
#     transform = nvisii.transform.create("table"),
#     material = nvisii.material.create("table")
# )
# table.get_transform().set_position((0.0,0.0,0.0))
# table_rot = nvisii.normalize(nvisii.quat(
#       0.5,
#       0.4999999999999999,
#       0.5,
#       0.5000000000000001,
#     ))
# table.get_transform().set_rotation(table_rot)
# table.get_transform().set_scale((0.013, 0.013, 0.013))
# # table.get_material().set_roughness(1)
# # table.get_material().set_base_color((0.1,0.1,0.1))

# mat_table = nvisii.material.get("table")

# mat_table.set_roughness(.5)

# # Lets set the base color and roughness of the object to use a texture. 
# # but the textures could also be used to set other
# # material propreties
# mat_table.set_base_color_texture(table_hsv)
# mat_table.set_roughness_texture(table_tex)

# # Set the collision with the floor mesh
# # first lets get the vertices 
# vertices = table.get_mesh().get_vertices()

# # get the position of the object
# pos = table.get_transform().get_position()
# pos = [pos[0],pos[1],pos[2]]
# scale = table.get_transform().get_scale()
# scale = [scale[0],scale[1],scale[2]]
# rot = table.get_transform().get_rotation()
# rot = [rot[0],rot[1],rot[2],rot[3]]

# # create a collision shape that is a convex hull
# obj_col_id = p.createCollisionShape(
#     p.GEOM_MESH,
#     vertices = vertices,
#     # fileName ="./tabletop_gym/envs/assets/objects/table.obj",
#     meshScale = scale,
# )

# # create a body without mass so it is static
# p.createMultiBody(
#     baseCollisionShapeIndex = obj_col_id,
#     basePosition = pos,
#     baseOrientation= rot,
# )    





# lets create a bunch of objects 
# mesh = nvisii.mesh.create_teapotahedron('mesh')
mesh = nvisii.mesh.create_from_file("obj", "./tabletop_gym/envs/assets/google_objects/green_tape/meshes/model.obj")
# set up for pybullet - here we will use indices for 
# objects with holes 
vertices = mesh.get_vertices()
indices = mesh.get_triangle_indices()

tex = nvisii.texture.create_from_file("tex",'./tabletop_gym/envs/assets/google_objects/green_tape/meshes/texture.png')

# Textures can be mixed and altered. 
# Checkout create_hsv, create_add, create_multiply, and create_mix
object_tex = nvisii.texture.create_hsv("tape", tex, 
    hue = 0, saturation = .5, value = 1.0, mix = 1.0)



ids_pybullet_and_nvisii_names = []

for i in range(opt.nb_objects):
    name = f"mesh_{i}"
    obj= nvisii.entity.create(
        name = name,
        transform = nvisii.transform.create(name),
        material = nvisii.material.create(name)
    )
    obj.set_mesh(mesh)
    mat = nvisii.material.get(name)

    mat.set_roughness(.5)

    # Lets set the base color and roughness of the object to use a texture. 
    # but the textures could also be used to set other
    # material propreties
    mat.set_base_color_texture(object_tex)
    mat.set_roughness_texture(tex)

    # transforms
    pos = nvisii.vec3(
        random.uniform(-0.4,0.4),
        random.uniform(-0.4,0.4),
        random.uniform(1.4,1.42)
    )
    rot = nvisii.normalize(nvisii.quat(
        random.uniform(-1,1),
        random.uniform(-1,1),
        random.uniform(-1,1),
        random.uniform(-1,1),
    ))
    s = random.uniform(0.9,1.1)
    scale = (s,s,s)

    obj.get_transform().set_position(pos)
    obj.get_transform().set_rotation(rot)
    obj.get_transform().set_scale(scale)

    # pybullet setup 
    pos = [pos[0],pos[1],pos[2]]
    rot = [rot[0],rot[1],rot[2],rot[3]]
    scale = [scale[0],scale[1],scale[2]]

    obj_col_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices = vertices,
        meshScale = scale,
        # if you have static object like a bowl
        # this allows you to have concave objects, but 
        # for non concave object, using indices is 
        # suboptimal, you can uncomment if you want to test
        # indices =  indices,  
    )
    
    p.createMultiBody(
        baseCollisionShapeIndex = obj_col_id,
        basePosition = pos,
        baseOrientation= rot,
        baseMass = random.uniform(0.5,2)
    )       

    # to keep track of the ids and names 
    ids_pybullet_and_nvisii_names.append(
        {
            "pybullet_id":obj_col_id, 
            "nvisii_id":name
        }
    )

    # Material setting
    rgb = colorsys.hsv_to_rgb(
        random.uniform(0,1),
        random.uniform(0.7,1),
        random.uniform(0.7,1)
    )

    # obj.get_material().set_base_color(rgb)

    # obj_mat = obj.get_material()
    # r = random.randint(0,2)

    # # This is a simple logic for more natural random materials, e.g.,  
    # # mirror or glass like objects
    # if r == 0:  
    #     # Plastic / mat
    #     obj_mat.set_metallic(0)  # should 0 or 1      
    #     obj_mat.set_transmission(0)  # should 0 or 1      
    #     obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
    # if r == 1:  
    #     # metallic
    #     obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
    #     obj_mat.set_transmission(0)  # should 0 or 1      
    # if r == 2:  
    #     # glass
    #     obj_mat.set_metallic(0)  # should 0 or 1      
    #     obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

    # if r > 0: # for metallic and glass
    #     r2 = random.randint(0,1)
    #     if r2 == 1: 
    #         obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
    #     else:
    #         obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  

# Lets run the simulation for a few steps. 
steps_per_frame = math.ceil( 1.0 / (seconds_per_step * frames_per_second) )
for j in range(steps_per_frame):
    p.stepSimulation()

# Lets update the pose of the objects in nvisii 
for ids in ids_pybullet_and_nvisii_names:

    # get the pose of the objects
    pos, rot = p.getBasePositionAndOrientation(ids['pybullet_id'])

    # get the nvisii entity for that object
    obj_entity = nvisii.entity.get(ids['nvisii_id'])
    obj_entity.get_transform().set_position(pos)

    # nvisii quat expects w as the first argument
    obj_entity.get_transform().set_rotation(rot)
print(f'rendering frame {str(i).zfill(5)}/{str(opt.nb_frames).zfill(5)}')

nvisii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"{opt.outf}/{str(i).zfill(5)}.png"
)

camera_2 = nvisii.entity.create(
    name = "camera_2",
    transform = nvisii.transform.create("camera_2"),
    camera = nvisii.camera.create_from_fov(
        name = "camera_2", 
        field_of_view = FOV/180 * np.pi,
        aspect = 1.0
    )
)
print(camera_coord())
camera_2.get_transform().look_at(
    at = (0.08, 0.02, 1.07),
    up = (0,0,1),
    eye = (-0.4, -0.5, 1.87),
)
nvisii.set_camera_entity(camera_2)

nvisii.render_to_file(
    width=int(opt.width), 
    height=int(opt.height), 
    samples_per_pixel=int(opt.spp),
    file_path=f"{opt.outf}/{str(i).zfill(5)}_2.png"
)

p.disconnect()
nvisii.deinitialize()

subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))