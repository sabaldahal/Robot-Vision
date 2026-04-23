import sys
path_to_blender_packages = "/Users/sabaldahal/Desktop/College/WORK-RESEARCH LAB/spacecraft blender/src/v2/Robot-Vision/local/blender_packages"
sys.path.append(path_to_blender_packages)
import os
current_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(current_path))



import bpy
from mathutils import Vector
from bpy import context
import numpy as np
import itertools
import cv2
from bpy_extras.object_utils import world_to_camera_view
import os
import time

from utils.config import Config
from utils.bbox import *
from utils.keypoints import *
from utils.randomizer import *
from utils.sdgdata import *
from utils.dataformatter import *
from utils.transformation_matrix import * 








### helper functions

def render(output_path):
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

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


### blender data access
scene = bpy.context.scene
camera = bpy.data.objects['RealSense Camera']
resx = Config.image_resolution_x
resy = Config.image_resolution_y
bottom_collection = bpy.data.collections.get('BOTTOM FACES') 
top_collection = bpy.data.collections.get('TOP FACES')       
keypoint_collection = bpy.data.collections.get('Keypoints')
obj_controller = bpy.data.objects.get('SpacecraftController')
lights = bpy.data.collections.get('Lights')
all_classes_collection = top_collection.children
all_keypoints_collection = keypoint_collection.children

### data generation initialization
data = SDGData(scene, camera, resx, resy, bottom_collection, top_collection, all_classes_collection, all_keypoints_collection, keypoint_collection, obj_controller, lights)
keypoint_handler = KeyPoints(data)
bbox_handler = BoundingBox(data)
scene_randomizer = Randomizer(data)
data_formatter = DataFormatter(data)
transformation_matrix_calculator = TransformationMatrix(data)

### output directories setup
dir = Config.export_path
base_dir = os.makedirs(dir, exist_ok=True)
image_dir = os.path.join(dir, "images")
label_dir = os.path.join(dir, "labels")
matrix_label_dir = os.path.join(dir, "transformation_matrices")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(matrix_label_dir, exist_ok=True)


### data generation customization
totalimages = Config.total_images_to_generate
image_index = Config.image_start_index
generated_images = 0
coco_annotation_file = os.path.join(image_dir, "_annotations.coco.json")
coco_data_writer = data_formatter.export_data_COCO(coco_annotation_file, Config.save_annotations_file_after_every_n_images)
next(coco_data_writer)

### scene randomization customization
if Config.camera_to_object_distance_range is not None:
    scene_randomizer.settings.cameraDistance = Config.camera_to_object_distance_range


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
    if(Config.export_transformation_matrices):
        t_matrix = transformation_matrix_calculator.calculateMatrix()
        data_formatter.export_transformation_matrix(matrix_label_dir, image_index, t_matrix)

    
    image_index += 1
    generated_images += 1
    totalimages = totalimages - 1

    elapsedtime = time.time() - starttime

    print(f"{generated_images}/{Config.total_images_to_generate} images generated")
    print(f"Render time: {rendertime:.3f} seconds")
    print(f"Render + data processing time: {elapsedtime:.3f} seconds")
    print("----------------------------------------------------------")

#save and close coco json file
try:
    coco_data_writer.send(True)
except StopIteration:
    print("Coco Generator Stopped")