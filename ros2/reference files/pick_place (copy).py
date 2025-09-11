#!/usr/bin/env python3


from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time
"""
This script uses a color/depth camera to get the arm to find objects and pick them up. For this
demo, the arm is placed to the left of the camera facing outward. When the end-effector is located
at x=0, y=-0.3, z=0.2 w.r.t. the 'wx200/base_link' frame, the AR tag should be clearly visible to
the camera. A small basket should also be placed in front of the arm.

To get started, open a terminal and type:

    ros2 launch interbotix_xsarm_perception xsarm_perception.launch.py robot_model:=wx200

Then change to this directory and type:

    python3 pick_place.py
"""

ROBOT_MODEL = 'vx300'
ROBOT_NAME = ROBOT_MODEL
REF_FRAME = 'camera_color_optical_frame'
ARM_TAG_FRAME = f'{ROBOT_NAME}/ar_tag_link'
ARM_BASE_FRAME = f'{ROBOT_NAME}/base_link'


def main():
    # Create a global node to serve as the backend for each API component
    global_node = create_interbotix_global_node()
    # Initialize the arm module along with the pointcloud and armtag modules
    bot = InterbotixManipulatorXS(
        robot_model=ROBOT_MODEL,
        robot_name=ROBOT_NAME,
        node=global_node,
    )

    armtag = InterbotixArmTagInterface(
        ref_frame=REF_FRAME,
        arm_tag_frame=ARM_TAG_FRAME,
        arm_base_frame=ARM_BASE_FRAME,
        node_inf=global_node,
    )

    # Start up the API
    robot_startup(global_node)

    # set initial arm and gripper pose
    bot.arm.go_to_sleep_pose()
    bot.gripper.release()

    # get the ArmTag pose
    result = armtag.find_ref_to_arm_base_transform()
    print(f'printed this from find ref: {result}')
    bot.arm.set_ee_pose_components(x=-0.485, y=-0.371, z=0.204, roll=-0.038, pitch=0.373, yaw=-2.488)
    result = armtag.find_ref_to_arm_base_transform()
    time.sleep(5)
    print(f'printed this from find ref: {result}')

    bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    bot.arm.go_to_sleep_pose()
    bot.gripper.grasp()
    robot_shutdown(global_node)


if __name__ == '__main__':
    main()
