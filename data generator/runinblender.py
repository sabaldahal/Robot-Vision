import sys
#mac PC
# sys.path.append("/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/local/blender_packages")
# sys.path.append("/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/data generator")

#ubuntu IRAS LAB
sys.path.append("/home/sabal/code/spacecraft blender/latest/python/blender_packages")
sys.path.append("/home/sabal/code/spacecraft blender/latest/robot vision/Robot-Vision/data generator")

#ubuntu
dir = "/home/sabal/code/spacecraft blender/latest/robot vision/Robot-Vision/local/test dataset"
#mac
# dir = '/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/local/working model/update_dec_4_2025/renders_1_3_2026'

EXPORT_TRANFORMATION_MATRIX = True

import bpy
from mathutils import Vector
from bpy import context
import numpy as np
import itertools
import cv2
from bpy_extras.object_utils import world_to_camera_view
import os
import time


import bbox
import keypoints
import randomizer
import sdgdata
import dataformatter
import transformation_matrix


import importlib
def reload_modules():
    importlib.reload(bbox)
    importlib.reload(keypoints)
    importlib.reload(randomizer)
    importlib.reload(sdgdata)
    importlib.reload(dataformatter)
    importlib.reload(transformation_matrix)
    
reload_modules()

from bbox import *
from keypoints import *
from randomizer import *
from sdgdata import *
from dataformatter import *
from transformation_matrix import *


scene = bpy.context.scene
camera = bpy.data.objects['RealSense Camera']
resx = 1280
resy = 720

bottom_collection = bpy.data.collections.get('BOTTOM FACES')    #remove this line for multi class support
top_collection = bpy.data.collections.get('TOP FACES')          #remove this line for multi class support

keypoint_collection = bpy.data.collections.get('Keypoints')
obj_controller = bpy.data.objects.get('SpacecraftController')
lights = bpy.data.collections.get('Lights')

#MULTI CLASS LABELING SET UP
# face_a = bpy.data.collections.get('FACE A')
# face_b = bpy.data.collections.get('FACE B')
# face_c = bpy.data.collections.get('FACE C')
# face_d = bpy.data.collections.get('FACE D')
# all_classes_collection = [face_a, face_b, face_c, face_d]

all_classes_collection = top_collection.children
all_keypoints_collection = keypoint_collection.children

###

data = SDGData(scene, camera, resx, resy, bottom_collection, top_collection, all_classes_collection, all_keypoints_collection, keypoint_collection, obj_controller, lights)
keypoint_handler = KeyPoints(data)
bbox_handler = BoundingBox(data)
scene_randomizer = Randomizer(data)
#scene_randomizer.settings.cameraDistance = (0.3, 0.8)
#scene_randomizer.settings.cameraBounds.Z = (0.93, 2)
data_formatter = DataFormatter(data)
transformation_matrix_calculator = TransformationMatrix(data)



def render(output_path):
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)




base_dir = os.makedirs(dir, exist_ok=True)
image_dir = os.path.join(dir, "images")
label_dir = os.path.join(dir, "labels")
matrix_label_dir = os.path.join(dir, "transformation_matrices")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(matrix_label_dir, exist_ok=True)


from contextlib import contextmanager
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)

TOTAL_IMAGES_TO_GENERATE = 420
totalimages = TOTAL_IMAGES_TO_GENERATE
image_index = 100
generated_images = 0
coco_annotation_file = os.path.join(image_dir, "_annotations.coco.json")
coco_data_writer = data_formatter.export_data_COCO(coco_annotation_file, 25)
next(coco_data_writer)

#temp close up shots
# scene_randomizer.settings.cameraDistance = (0.3, 0.8)
# scene_randomizer.settings.cameraBounds.Z = (0.93, 2)
while totalimages > 0: 
    starttime = time.time()       
    image_path = os.path.join(image_dir, f"{image_index:06d}.png")

    scene_randomizer.randomize_camera_object_position()
    scene_randomizer.randomize_lights()
    bpy.context.view_layer.update()   
    keypointsData = keypoint_handler.project_keypoints_to_2D_from_collection()
    bboxData = bbox_handler.project_bbox_to_2D_from_collection()

    #check if at least 3 keypoints are visible
    bbcheck, kpcheck = data_formatter.filter_objects(bboxData, keypointsData)
    if len(kpcheck) == 0:
        continue
    # visible_count_total = 0
    # for kpt_list in keypointsData.values():
    #     visible_count_total += sum(1 for kp in kpt_list if kp["occluded"] == False)
    # if visible_count_total < 4:
    #     continue
    # visible_count = sum(1 for kp in keypointsData if kp["occluded"] == False)
    # if visible_count < 3:
    #     continue
    

    print("rendering...")
    renderstarttime = time.time()
    with stdout_redirected():
        render(image_path)
    rendertime = time.time() - renderstarttime
    #temp visualization
    #---------------------------
    # f_bbox, f_keypoints = data_formatter.filter_objects(bboxData, keypointsData)
    # keypoint_handler.draw_keypoints(image_path, f_keypoints)
    # bbox_handler.draw_bbox(image_path, f_bbox)
    #---------------------------

    #data_formatter.export_data_YOLO(label_dir, image_index, bboxData, keypointsData)
    coco_data_writer.send((image_index, bboxData, keypointsData))
    if(EXPORT_TRANFORMATION_MATRIX):
        t_matrix = transformation_matrix_calculator.calculateMatrix()
        data_formatter.export_transformation_matrix(matrix_label_dir, image_index, t_matrix)

    
    image_index += 1
    generated_images += 1
    totalimages = totalimages - 1

    elapsedtime = time.time() - starttime

    print(f"{generated_images}/{TOTAL_IMAGES_TO_GENERATE} images generated")
    print(f"Render time: {rendertime:.3f} seconds")
    print(f"Render + data processing time: {elapsedtime:.3f} seconds")
    print("----------------------------------------------------------")

#save and close coco json file
try:
    coco_data_writer.send(True)
except StopIteration:
    print("Coco Generator Stopped")