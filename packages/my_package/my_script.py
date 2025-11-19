#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from duckietown_msgs.msg import WheelsCmdStamped


class LaneFollowerNode:
    def __init__(self):
        rospy.loginfo("🔥 Advanced LaneFollowerNode starting up")

        # Vehicle name (namespace), default: duckiebot5
        self.veh = rospy.get_param("~veh", "duckiebot5")
        rospy.loginfo(f"Using vehicle name: {self.veh}")

        self.bridge = CvBridge()

        # Publishers
        self.pub_cmd = rospy.Publisher(
            f"/{self.veh}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1
        )

        self.pub_debug = rospy.Publisher(
            f"/{self.veh}/debug_image",
            Image,
            queue_size=1
        )

        # Subscribers
        rospy.Subscriber(
            f"/{self.veh}/camera_node/image/raw",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2**24
        )

        # Controller gains (tune if needed)
        self.Kp = rospy.get_param("~Kp", 0.65)
        self.Kd = rospy.get_param("~Kd", 0.12)
        self.base_speed = rospy.get_param("~base_speed", 0.28)

        self.last_error = 0.0
        self.green_go = False  # Traffic light state

    # ========== IMAGE PROCESSING ==========

    def preprocess(self, frame):
        """
        Convert to HSV, segment yellow lane, then compute Canny edges.
        Returns:
            edges (uint8 0/255)
            mask (yellow mask)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow lane HSV bounds (tune if needed)
        lower_yellow = np.array([15, 60, 60])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Canny edges over the mask
        edges = cv2.Canny(mask, 100, 200)

        return edges, mask

    def compute_lane_center_poly(self, edges):
        """
        Detect lane pixels in edges, fit polynomial x = a*y^2 + b*y + c,
        and compute lane center at the bottom of the image.
        Returns:
            cx (int): x-coordinate of lane center
            debug (BGR image): visualization of polynomial fit
        """
        h, w = edges.shape

        ys, xs = np.where(edges > 0)  # lane pixels

        if len(xs) < 50:
            # Not enough points, fallback: image center
            debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return w // 2, debug

        try:
            # Fit polynomial x = a*y^2 + b*y + c
            poly = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError:
            debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return w // 2, debug

        # Evaluate polynomial at bottom row
        y_eval = h - 1
        cx = poly[0] * y_eval * y_eval + poly[1] * y_eval + poly[2]

        # Debug visualization
        debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for y in range(0, h, 5):
            x = int(poly[0] * y * y + poly[1] * y + poly[2])
            if 0 <= x < w:
                cv2.circle(debug, (x, y), 2, (0, 255, 255), -1)

        return int(cx), debug

    # ========== RED STOP LINE DETECTION ==========

    def detect_red_stop_line(self, frame):
        """
        Detect a red stop line near the bottom of the image.
        Returns True if a strong red band is present.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red can wrap around hue=0, so we use two ranges
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        h, w = mask.shape
        # Look only near the bottom (stop line location)
        crop = mask[int(h * 0.75):h, :]

        red_pixels = np.sum(crop > 0)

        # Threshold may need tuning depending on lighting
        return red_pixels > 300

    # ========== TRAFFIC LIGHT DETECTION ==========

    def detect_traffic_light(self, frame):
        """
        Detect a green traffic light at the top-center of the image.
        Returns True if green is detected.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([45, 60, 60])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        h, w = mask.shape
        # Top-center region
        roi = mask[0:int(h * 0.25), int(w * 0.25):int(w * 0.75)]

        green_pixels = np.sum(roi > 0)

        # Threshold may need tuning
        return green_pixels > 200

    # ========== CONTROL HELPERS ==========

    def send_stop(self, t=0.0):
        """
        Publish zero velocity, optionally holding for t seconds.
        """
        cmd = WheelsCmdStamped()
        cmd.vel_left = 0.0
        cmd.vel_right = 0.0
        self.pub_cmd.publish(cmd)
        if t > 0:
            rospy.sleep(t)

    def drive_with_pd(self, cx, width):
        """
        Compute PD steering based on lane center position cx.
        width: image width.
        """
        # Normalized error in [-1, 1]
        error = (cx - (width / 2.0)) / (width / 2.0)
        d_error = error - self.last_error
        self.last_error = error

        omega = -(self.Kp * error + self.Kd * d_error)

        v = self.base_speed
        vl = v + omega
        vr = v - omega

        cmd = WheelsCmdStamped()
        cmd.vel_left = vl
        cmd.vel_right = vr
        self.pub_cmd.publish(cmd)

    def publish_debug(self, debug_img):
        if self.pub_debug.get_num_connections() == 0:
            return
        msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        self.pub_debug.publish(msg)

    # ========== MAIN CALLBACK ==========

    def callback(self, msg):
        """
        Main camera callback:
        1. Wait for green light before starting.
        2. Stop at red stop line.
        3. Follow lane using polynomial fit + PD.
        """
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame.shape

        # 1) Traffic light: wait until green
        if not self.green_go:
            if self.detect_traffic_light(frame):
                rospy.loginfo("🟢 GREEN LIGHT detected — starting navigation")
                self.green_go = True
            else:
                rospy.loginfo_throttle(2.0, "🔴 Waiting for green light...")
                self.send_stop()
                return

        # 2) Red stop line detection
        if self.detect_red_stop_line(frame):
            rospy.loginfo("🟥 Red stop line detected — stopping briefly")
            self.send_stop(t=2.0)
            # After stop, continue (do not return) so we can roll forward again
            # Optionally you could toggle a state here for intersection logic.

        # 3) Lane processing
        edges, _ = self.preprocess(frame)
        cx, debug = self.compute_lane_center_poly(edges)

        # 4) Control
        self.drive_with_pd(cx, w)

        # 5) Debug image
        # Draw a vertical line at cx for visualization
        cv2.line(debug, (cx, 0), (cx, h - 1), (0, 0, 255), 2)
        self.publish_debug(debug)


if __name__ == "__main__":
    rospy.init_node("lane_follower_node")
    node = LaneFollowerNode()
    rospy.spin()