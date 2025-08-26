import sys
#mac PC
sys.path.append("/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/practice/v2/python/blender_packages")
sys.path.append("/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/practice/v2/python")

#ubuntu IRAS LAB
#sys.path.append("/home/sabal/code/spacecraft blender/latest/python/blender_packages")
#sys.path.append("/home/sabal/code/spacecraft blender/latest/python")

import bpy
from mathutils import Vector
from bpy import context
import numpy as np
import itertools
import cv2
from bpy_extras.object_utils import world_to_camera_view
import os


import bbox
import keypoints
import randomizer
import sdgdata
import dataformatter


import importlib
def reload_modules():
    importlib.reload(bbox)
    importlib.reload(keypoints)
    importlib.reload(randomizer)
    importlib.reload(sdgdata)
    importlib.reload(dataformatter)
    
reload_modules()

from bbox import *
from keypoints import *
from randomizer import *
from sdgdata import *
from dataformatter import *


scene = bpy.context.scene
camera = bpy.data.objects['RealSense Camera']
resx = 1280
resy = 720
bottom_collection = bpy.data.collections.get('BOTTOM FACES')
top_collection = bpy.data.collections.get('TOP FACES')
keypoint_collection = bpy.data.collections.get('Keypoints')
obj_controller = bpy.data.objects.get('SpacecraftController')
lights = bpy.data.collections.get('Lights')

data = SDGData(scene, camera, resx, resy, bottom_collection, top_collection, keypoint_collection, obj_controller, lights)
keypoint_handler = KeyPoints(data)
bbox_handler = BoundingBox(data)
scene_randomizer = Randomizer(data)
data_formatter = DataFormatter(data)


def render(output_path):
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


dir = "/home/sabal/code/spacecraft blender/latest/blenderRender/testrenders"

base_dir = os.makedirs(dir, exist_ok=True)
image_dir = os.path.join(dir, "images")
label_dir = os.path.join(dir, "labels")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)


totalimages = 3
image_index = 0
coco_annotation_file = os.path.join(image_dir, "_annotations.coco.json")
coco_data_writer = data_formatter.export_data_COCO(coco_annotation_file, 100)
next(coco_data_writer)
while totalimages > 0:        
    image_path = os.path.join(image_dir, f"{image_index:06d}.png")
    scene_randomizer.randomize_camera_object_position()
    scene_randomizer.randomize_lights()
    bpy.context.view_layer.update()   
    keypointsData = keypoint_handler.project_keypoints_to_2D()
    #only continue if at least 3 keypoints are visible
    visible_count = sum(1 for kp in keypointsData if kp["occluded"] == False)
    if visible_count < 3:
        continue
    
    bboxData = bbox_handler.project_bbox_to_2D()
    render(image_path)
    #keypoint_handler.draw_keypoints(image_path, keypointsData)
    #bbox_handler.draw_bbox(image_path, bboxData)
    data_formatter.export_data_YOLO(label_dir, image_index, bboxData, keypointsData)
    coco_data_writer.send((image_index, bboxData, keypointsData))
    print(f"{image_index+1} images generated")
    image_index += 1
    totalimages = totalimages - 1

#save and close coco json file
try:
    coco_data_writer.send(True)
except StopIteration:
    print("Coco Generator Stopped")