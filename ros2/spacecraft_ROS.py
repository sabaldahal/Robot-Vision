#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from ultralytics import YOLO
import numpy as np
import time
import json
import cv2 as cv

# Robot Configurations
ROBOT_MODEL = 'vx300'
ROBOT_NAME = ROBOT_MODEL
REF_FRAME = 'camera_color_optical_frame'
ARM_TAG_FRAME = f'{ROBOT_NAME}/ar_tag_link'
ARM_BASE_FRAME = f'{ROBOT_NAME}/base_link'

MODEL_PATH = "./weights/best.pt"

OBJ_POINTS = []

coords_file = "./model/coords.json"
keypointsArr = []

with open (coords_file, "r") as f:
    keypointsArr = json.load(f)

for k in keypointsArr:
    OBJ_POINTS.append(k['location'])

OBJ_POINTS = np.array(OBJ_POINTS, dtype=np.float32)

fx = 915.5166015625
fy = 915.607421875
cx = 629.287109375
cy = 356.802307128906


CAM_MAT = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

class Spacecraft_ROS(Node):
    def __init__(self, bot, armtag):
        super().__init__('spacecraft_ros')
        self.bot: InterbotixManipulatorXS = bot
        self.armtag: InterbotixArmTagInterface = armtag
        self.detections = []
        self.model = YOLO(MODEL_PATH)

        # YOLO Detections Subscriber
        self.subscription = self.create_subscription(
            DetectionArray,
            '/camera/camera/color/image_raw',
            self.yolo_callback,
            10
        )
        self.get_logger().info("Listening for YOLO detections...")

    def yolo_callback(self, msg):
        """Callback to process YOLO detections and transform coordinates."""
        self.detections.clear()  # Clear previous detections

        # Retrieve the camera-to-arm transform
        self.armtag.find_ref_to_arm_base_transform()
        camera_base_trans = self.armtag.get_transform(
            self.armtag.tfBuffer, 
            target_frame=ARM_BASE_FRAME, 
            source_frame=REF_FRAME
        )

        result = self.model(msg.img)[0]
        keypoints = result.keypoints.xy.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()
        img_points = []
        for kps in keypoints:
            for x,y in kps:
                img_points.append([x,y])
        
        success, rvec, tvec = cv.solvePnp(
            OBJ_POINTS,
            img_points,
            CAM_MAT,
            DIST_COEFFS
        )

        if success:
            Rotation, _ = cv.Rodrigues(rvec)
            Translation = np.asarray(tvec).reshape(3,1)



        for detection in msg.detections:
            class_name = detection.class_name
            bbox3d = detection.bbox3d
            x_cam, y_cam, z_cam = bbox3d.center.position.x, bbox3d.center.position.y, bbox3d.center.position.z

            # Transform coordinates
            #camera axis convention: x:right, y:down, z:forward
            #robot axis convention: x:forward, y:left, z:up
            #run this to get the camera_color_optical_frame_extrinsics
            #ros2 run tf2_ros tf2_echo camera_link camera_color_optical_frame
            #it will show the translation from camera_link. camera_link is the origin of the camera
            
            xyz_unaligned = np.array([[0.015 - y_cam], [-z_cam], [x_cam], [1]])
            xyz_aligned = np.matmul(camera_base_trans, xyz_unaligned)

            R_swap = np.array([
                [0, -1, 0],
                [0,  0, -1],
                [1,  0, 0]
            ])

            R_obj_unaligned = R_swap @ Rotation
            R_cam_to_base = camera_base_trans[:3, :3]
            R_obj_in_base = R_cam_to_base @ R_obj_unaligned


            # Append transformed coordinates
            self.detections.append({
                'class_name': class_name,
                'x': round(xyz_aligned[0, 0], 3),
                'y': round(xyz_aligned[1, 0], 3),
                'z': round(xyz_aligned[2, 0], 3),
                'rotation': R_obj_in_base
            })
        self.get_logger().info("YOLO Detections Transformed and Ready.")

    def move_robot(self):
        """Pick and place objects based on YOLO detections."""
        if not self.detections:
            self.get_logger().warn("No YOLO detections available.")
            return

        for detection in self.detections:
            x, y, z = detection['x'], detection['y'], detection['z']
            rotation = detection['rotation']
            self.get_logger().info(f"Moving robot '{detection['class_name']}' at x={x}, y={y}, z={z}")

            # Move arm above the object
            self.bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.1, pitch=0.5)
            self.bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.02, pitch=0.5)
            
            # Grasp the object
            self.bot.gripper.grasp()
            self.bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.1, pitch=0.5)

            # Place the object in a predefined location
            self.bot.arm.set_ee_pose_components(x=-0.35, y=-0.2, z=0.2)
            self.bot.gripper.release()

def main():
    rclpy.init()
    global_node = create_interbotix_global_node()

    try:
        # Initialize robot interfaces
        bot = InterbotixManipulatorXS(robot_model=ROBOT_MODEL, robot_name=ROBOT_NAME, node=global_node)
        armtag = InterbotixArmTagInterface(
            ref_frame=REF_FRAME, 
            arm_tag_frame=ARM_TAG_FRAME, 
            arm_base_frame=ARM_BASE_FRAME, 
            node_inf=global_node
        )
        yolo_node = Spacecraft_ROS(bot, armtag)

        # Start robot
        robot_startup(global_node)

        # Set initial arm and gripper pose
        bot.arm.go_to_home_pose()
        bot.arm.go_to_sleep_pose()
        bot.gripper.release()
        yolo_node.armtag.find_ref_to_arm_base_transform()
        bot.arm.set_ee_pose_components(x=0.3, z=0.2)

        # Continuous loop for checking and picking objects
        last_detection_time = time.time()  # Record current time

        while rclpy.ok():  # Loop until the program is stopped
            # Wait for a single detection
            while not yolo_node.detections and rclpy.ok():
                rclpy.spin_once(yolo_node, timeout_sec=0.5)
            # Check if 300 seconds have passed since last detection
                if time.time() - last_detection_time > 300:
                    print("\n--- No Objects Detected for 300 Seconds. Exiting Program. ---")
                    return  # Exit the program
                
            if yolo_node.detections:
                print("\n--- New Object Detected! ---")
                yolo_node.move_robot() 

                # Update last detection time
                last_detection_time = time.time()

                # Clear detections after processing to wait for the next object
                yolo_node.detections.clear()
                print("\n--- Object Processed. Waiting for Next Object... ---")

    finally:
        bot.arm.go_to_home_pose()
        bot.arm.go_to_sleep_pose()
        robot_shutdown(global_node)
        if rclpy.ok():
            rclpy.shutdown()
        print("Pick and Place Node Shutdown.")

if __name__ == '__main__':
    main()