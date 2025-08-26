import bpy
from mathutils import Vector
from bpy import context
import numpy as np
import itertools
import cv2
from bpy_extras.object_utils import world_to_camera_view
import os


class BoundingBox():
    def __init__(self, data):
        self.data = data

    def draw_bbox(self, output_path, bbox):
        # Load image using OpenCV
        img = cv2.imread(output_path)
        if img is None:
            print(f"Could not load image: {output_path}")
            return

        # Draw bounding box
        color = (0, 0, 255)  # Red
        thickness = 1
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Save modified image
        base, ext = os.path.splitext(output_path)
        out_path = base + "_bbox.png"
        cv2.imwrite(out_path, img)
        print(f"Saved with bounding box: {out_path}")

    def raycast_detect_corners_obj(self, obj):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()

        world_matrix = obj.matrix_world
        camera_matrix = self.data.camera.matrix_world.inverted()

        screen_coords = []

        for vertex in mesh.vertices:
            world_co = world_matrix @ vertex.co
            camera_co = camera_matrix @ world_co

            if camera_co.z < 0:
                screen_co = world_to_camera_view(self.data.scene, self.data.camera, world_co)
                px = screen_co.x * self.data.resx
                py = (1-screen_co.y) * self.data.resy
                screen_coords.append((px, py))
        return screen_coords

    def raycast_detect_corners_collection(self):
        screen_coords = []
        for a in self.data.bottom_collection.all_objects:
            screen_coords.extend(self.raycast_detect_corners_obj(a))
        for b in self.data.top_collection.all_objects:
            screen_coords.extend(self.raycast_detect_corners_obj(b))

        if screen_coords:
            xs, ys = zip(*screen_coords)
            bbox = (min(xs), min(ys), max(xs), max(ys))
            return bbox
        
        return None

    def project_bbox_to_2D(self):
        return self.raycast_detect_corners_collection()