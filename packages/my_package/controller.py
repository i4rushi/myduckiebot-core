import time
import numpy as np
from perception import detect_lane_markings, detect_light, detect_stop_line, detect_sign
from utils import get_steer_matrix_left_lane_markings, get_steer_matrix_right_lane_markings

class LaneServoController:
    def __init__(self, img_shape):
        h, w, _ = img_shape
        self.steer_left = get_steer_matrix_left_lane_markings((h, w))
        self.steer_right = get_steer_matrix_right_lane_markings((h, w))

        self.k_yellow = 0.9
        self.k_white = 0.9
        self.turn_gain_to_wheels = 0.5
        self.base_speed = 0.30

        # State variables for traffic light
        self.traffic_light_state = "NORMAL"
        self.traffic_start_time = 0

        # State variables for stop sign handling
        self.stop_sign_state = "NORMAL"  # NORMAL, STOPPING, CROSSING
        self.stop_start_time = 0
        self.cross_start_time = 0

        # State variables for slow sign handling
        self.slow_sign_state = "NORMAL"  # NORMAL, SLOWING
        self.slow_start_time = 0
        self.slow_speed = 0.15  # Reduced speed for slow sign

    def compute_action(self, obs_rgb):
        mask_left, mask_right = detect_lane_markings(obs_rgb)

        def mean_weight(m, s):
            total = m.sum()
            return 0.0 if total < 1e-6 else float((m * s).sum() / total)

        left_mean = mean_weight(mask_left, self.steer_left)
        right_mean = mean_weight(mask_right, self.steer_right)

        omega = self.k_yellow * left_mean - self.k_white * right_mean
        omega = np.clip(omega, -1.0, 1.0)

        speed = self.base_speed * (1.0 - 0.5 * abs(omega))
        speed = float(np.clip(speed, 0.05, 0.8))

        left_wheel  = speed - self.turn_gain_to_wheels * omega
        right_wheel = speed + self.turn_gain_to_wheels * omega
        return np.array([np.clip(left_wheel, -1, 1),
                         np.clip(right_wheel, -1, 1)], dtype=np.float32)

    def handle_traffic_light(self, obs, action):
        """
        Handle traffic light state machine
        Returns modified action and new state
        """
        current_time = time.time()
        light = detect_light(obs)
        stop_line_detected = detect_stop_line(obs)

        if self.traffic_light_state == "NORMAL":
            if light == "RED" and stop_line_detected:
                print("Traffic light is red - stopping until light is green")
                self.traffic_light_state = "WAITING"
                return np.array([0.0, 0.0]), self.traffic_light_state

        elif self.traffic_light_state == "WAITING":
            if light == "GREEN":
                print("Green light - resumung")
                self.traffic_light_state = "NORMAL"
                return np.array([self.base_speed, self.base_speed]), self.traffic_light_state
            else:
                return np.array([0.0, 0.0]), self.traffic_light_state

        return action, self.traffic_light_state

    def handle_stop_sign(self, obs, action):
        """
        Handle stop sign state machine
        Returns modified action and new state
        """
        current_time = time.time()

        if self.stop_sign_state == "NORMAL":
            # Check if we see a stop sign and are at a stop line
            sign = detect_sign(obs)
            stop_line_detected = detect_stop_line(obs)

            if sign == "STOP" and stop_line_detected:
                print("Stop sign detected at stop line - stopping for 3 seconds")
                self.stop_sign_state = "STOPPING"
                self.stop_start_time = current_time
                return np.array([0.0, 0.0]), self.stop_sign_state

        elif self.stop_sign_state == "STOPPING":
            # Stop for 3 seconds
            if current_time - self.stop_start_time < 3.0:
                return np.array([0.0, 0.0]), self.stop_sign_state
            else:
                print("3 seconds passed - proceeding through intersection")
                self.stop_sign_state = "CROSSING"
                self.cross_start_time = current_time
                # Go straight
                return np.array([self.base_speed, self.base_speed]), self.stop_sign_state

        elif self.stop_sign_state == "CROSSING":
            # Go straight for 2 seconds
            if current_time - self.cross_start_time < 2.0:
                return np.array([self.base_speed, self.base_speed]), self.stop_sign_state
            else:
                print("Intersection crossed - resuming normal operation")
                self.stop_sign_state = "NORMAL"

        return action, self.stop_sign_state

    def handle_slow_sign(self, obs, action):
        """
        Handle slow sign state machine
        Returns modified action and new state
        """
        current_time = time.time()

        if self.slow_sign_state == "NORMAL":
            # Check if we see a slow sign
            sign = detect_sign(obs)

            if sign == "SLOW":
                print("Slow sign detected - reducing speed for 3 seconds")
                self.slow_sign_state = "SLOWING"
                self.slow_start_time = current_time
                # Reduce speed but keep moving forward
                slow_action = self.compute_slow_action(obs)
                return slow_action, self.slow_sign_state

        elif self.slow_sign_state == "SLOWING":
            # Move slowly for 3 seconds
            if current_time - self.slow_start_time < 3.0:
                slow_action = self.compute_slow_action(obs)
                return slow_action, self.slow_sign_state
            else:
                print("3 seconds passed - resuming normal speed")
                self.slow_sign_state = "NORMAL"

        return action, self.slow_sign_state

    def compute_slow_action(self, obs_rgb):
        """
        Compute action with reduced speed for slow sign
        """
        mask_left, mask_right = detect_lane_markings(obs_rgb)

        def mean_weight(m, s):
            total = m.sum()
            return 0.0 if total < 1e-6 else float((m * s).sum() / total)

        left_mean = mean_weight(mask_left, self.steer_left)
        right_mean = mean_weight(mask_right, self.steer_right)

        omega = self.k_yellow * left_mean - self.k_white * right_mean
        omega = np.clip(omega, -1.0, 1.0)

        # Use reduced speed for slow sign
        speed = self.slow_speed * (1.0 - 0.5 * abs(omega))
        speed = float(np.clip(speed, 0.05, self.slow_speed))

        left_wheel  = speed - self.turn_gain_to_wheels * omega
        right_wheel = speed + self.turn_gain_to_wheels * omega
        return np.array([np.clip(left_wheel, -1, 1),
                         np.clip(right_wheel, -1, 1)], dtype=np.float32)