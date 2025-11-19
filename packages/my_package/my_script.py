#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from turbojpeg import TurboJPEG

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage

import os

HOST_NAME = os.environ["VEHICLE_NAME"]

# HSV mask for yellow lane
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False


class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.veh = HOST_NAME
        self.jpeg = TurboJPEG()

        # PID control variables
        self.offset = 220        # image center offset
        self.velocity = 0.34
        self.speed = 0.6

        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.proportional = None

        self.P = 0.049
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()

        # ROS communication
        if DEBUG:
            self.pub = rospy.Publisher(f"/{self.veh}/output/image/mask/compressed",
                                       CompressedImage, queue_size=1)

        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed",
                                    CompressedImage, self.callback,
                                    queue_size=1, buff_size="20MB")

        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )

        self.loginfo("Lane Following Node Initialized")

    # ----------------------------------------------------------------------
    # PROCESS CAMERA CALLBACK
    # ----------------------------------------------------------------------
    def callback(self, msg):
        img = self.jpeg.decode(msg.data)

        # Crop lower image (road area)
        crop = img[300:, :, :]
        crop_width = crop.shape[1]

        # Convert to HSV and mask yellow lane
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])

        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        # Find largest contour as lane line
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            # Compute centroid for steering error
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                self.proportional = cx - (crop_width // 2) + self.offset
            except:
                self.proportional = 0
        else:
            # If no lane found, assume drifting right
            self.proportional = -100

        # Publish debug image
        if DEBUG:
            debug_msg = CompressedImage(
                format="jpeg",
                data=self.jpeg.encode(crop)
            )
            self.pub.publish(debug_msg)

    # ----------------------------------------------------------------------
    # PID + DRIVING
    # ----------------------------------------------------------------------
    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
            self.twist.v = self.velocity
            self.vel_pub.publish(self.twist)
            return

        # P term
        P_term = -self.proportional * self.P

        # D term
        dt = rospy.get_time() - self.last_time
        d_error = (self.proportional - self.last_error) / dt
        D_term = d_error * self.D

        self.last_error = self.proportional
        self.last_time = rospy.get_time()

        self.twist.v = self.velocity
        self.twist.omega = P_term + D_term

        if DEBUG:
            self.loginfo(f"P={P_term}, D={D_term}, Omega={self.twist.omega}")

        self.vel_pub.publish(self.twist)

    # ----------------------------------------------------------------------
    # SHUTDOWN
    # ----------------------------------------------------------------------
    def hook(self):
        self.twist.v = 0
        self.twist.omega = 0
        for _ in range(8):
            self.vel_pub.publish(self.twist)
        print("SHUTTING DOWN")


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node_clean")
    rate = rospy.Rate(8)

    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
