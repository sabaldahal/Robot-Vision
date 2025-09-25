import sys

import bpy
from mathutils import Vector, Euler, Quaternion
from bpy import context
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
import random
import math
from typing import Tuple

class Bounds():
    def __init__(self, x:Tuple[float, float], y:Tuple[float, float], z:Tuple[float, float]):
        self.X = x
        self.Y = y
        self.Z = z


class RandomizerSettings():
    def __init__(self):
        self.objectBounds = None
        self.cameraBounds = None
        self.changeObjectPositionX = True
        self.changeObjectPositionY = True
        self.changeObjectPositionZ = False
        self.changeCameraPositionX = True
        self.changeCameraPositionY = True
        self.changeCameraPositionZ = True
        self.rotateObjectX = False
        self.rotateObjectY = False
        self.rotateObjectZ = True


class Randomizer():
    def __init__(self, data, settings = None):
        self.data = data
        if settings==None:
            self.settings = RandomizerSettings()
            self.settings.objectBounds = Bounds(x=(-2.0, 2.0), y=(-1.215, 1.215), z=(0.93, 2.0))
            self.settings.cameraBounds = Bounds(x=(-2.0, 2.0), y=(-1.215, 1.215), z=(0.93, 2.0))
        else:
            self.settings = settings

    def randomize_camera_rotation(self, max_degrees=6):
        camera = self.data.camera
        max_radians = math.radians(max_degrees)
        max_radiansz = math.radians(3)

        #camera.rotation_euler[0] += random.uniform(-max_radians, max_radians)  # X (pitch)
        camera.rotation_euler[1] += random.uniform(-max_radians, max_radians)  # Y (roll)
        camera.rotation_euler[2] += random.uniform(-max_radians, max_radians) 

    def set_minimum_distance(self, minDistance=0.25):
        distance = (self.data.camera.location - self.data.obj_controller.location).length
        if(distance < 0.25):
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
        objBoundsX = self.settings.objectBounds.X
        objBoundsY = self.settings.objectBounds.Y
        objBoundsZ = self.settings.objectBounds.Z
        camBoundsX = self.settings.cameraBounds.X
        camBoundsY = self.settings.cameraBounds.Y
        camBoundsZ = self.settings.cameraBounds.Z

        rotation = (0, 360)
        reduce = 0.066
        reducedBoundsX = (objBoundsX[0]+reduce, objBoundsX[1] - reduce)
        reducedBoundsY = (objBoundsY[0]+reduce, objBoundsY[1] - reduce)
        #random object orientation and position
        obj = self.data.obj_controller
        objx = obj.location.x
        objy = obj.location.y
        objz = obj.location.z
        objRx = 0
        objRy = 0
        objRz = 0
        #change coordinates
        if self.settings.changeObjectPositionX: objx = random.uniform(*reducedBoundsX)
        if self.settings.changeObjectPositionY: objy = random.uniform(*reducedBoundsY)
        if self.settings.changeObjectPositionZ: objz = random.uniform(*objBoundsZ)

        if self.settings.rotateObjectX: objRx = random.uniform(*rotation)
        if self.settings.rotateObjectY: objRy = random.uniform(*rotation)
        if self.settings.rotateObjectZ: objRz = random.uniform(*rotation)
        
        obj.location = Vector((objx, objy, objz))
        obj.rotation_euler = (math.radians(objRx), math.radians(objRy), math.radians(objRz))



        #random camera position
        camx = random.uniform(*camBoundsX)
        camy = random.uniform(*camBoundsY)
        camz = random.uniform(*camBoundsZ)
        
        if random.random() > 0.15:
            self.data.camera.location = Vector((camx, camy, camz))
        self.lookAtObject()
        distance, minDistance = self.set_minimum_distance()
        offsetVal = 0.05
        if distance < minDistance:
            offsetVal = 0.015
        self.offset_camera_position(offsetVal)
        #random camera rotation
        self.randomize_camera_rotation()

    def randomize_lights(self):
        energyR = (1, 8)
        lightProperty = self.data.lights.objects[0].data
        lightProperty.energy = random.uniform(*energyR)
        for p in self.data.lights.objects:
                p.hide_render = random.random() < 0.15





