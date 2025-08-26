import sys

import bpy
from mathutils import Vector, Euler, Quaternion
from bpy import context
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
import random
import math


class Randomizer():
    def __init__(self, data):
        self.data = data

    def randomize_camera_rotation(self, max_degrees=10):
        camera = self.data.camera
        max_radians = math.radians(max_degrees)
        max_radiansz = math.radians(3)

        #camera.rotation_euler[0] += random.uniform(-max_radians, max_radians)  # X (pitch)
        camera.rotation_euler[1] += random.uniform(-max_radians, max_radians)  # Y (roll)
        camera.rotation_euler[2] += random.uniform(-max_radians, max_radians) 

    def set_minimum_distance(self, minDistance=0.52):
        distance = (self.data.camera.location - self.data.obj_controller.location).length
        if(distance < 0.45):
            d = (self.data.camera.location - self.data.obj_controller.location).normalized()
            self.data.camera.location = self.data.obj_controller.location + d * minDistance
        return (distance, minDistance)
    
    def offset_camera_position(self, offsetVal=0.2):
        offset = offsetVal   
        ox = random.uniform(-offset, offset)
        oy= random.uniform(-offset, offset)        
        offsetVector = Vector((ox, oy, 0))
        self.data.camera.location = self.data.camera.location + offsetVector
        
    def lookAtObject(self):
        direction = self.data.obj_controller.location - self.data.camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.data.camera.rotation_euler = rot_quat.to_euler()

    def randomize_camera_object_position(self):
        #bounds
        tablex = (-2.0, 2.0)
        tabley = (-1.215, 1.215)
        tablez_exp_max_values = [2.68, 1.93, 2.2]
        tablez = (0.93, 2)
        rotation = (0, 360)
        reduce = 0.066
        smalltablex = (tablex[0]+reduce, tablex[1] - reduce)
        smalltabley = (tabley[0]+reduce, tabley[1] - reduce)
        #random object orientation with positionz fixed and rotation xy fixed
        obj = self.data.obj_controller
        objx = random.uniform(*smalltablex)
        objy = random.uniform(*smalltabley)
        objRz = random.uniform(*rotation)
        obj.location = Vector((objx, objy, obj.location.z))
        obj.rotation_euler = (math.radians(0), math.radians(0), math.radians(objRz))

        #random camera position
        camx = random.uniform(*tablex)
        camy = random.uniform(*tabley)
        camz = random.uniform(*tablez)
        
        if random.random() > 0.15:
            self.data.camera.location = Vector((camx, camy, camz))
        self.lookAtObject()
        distance, minDistance = self.set_minimum_distance()
        offsetVal = 0.2
        if distance < minDistance:
            offsetVal = 0.07
        self.offset_camera_position(offsetVal)
        #random camera rotation
        self.randomize_camera_rotation()

    def randomize_lights(self):
        energyR = (1, 8)
        lightProperty = self.data.lights.objects[0].data
        lightProperty.energy = random.uniform(*energyR)
        for p in self.data.lights.objects:
                p.hide_render = random.random() < 0.15





