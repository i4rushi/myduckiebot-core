#!/usr/bin/env python3
"""
Lane keeping in gym-duckietown using visual servoing,
with optional debug overlay for lane detection.
"""

import argparse
import time
import numpy as np
import cv2

from gym_duckietown.simulator import Simulator
from controller import LaneServoController
from perception import detect_lane_markings
from utils import make_debug_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="udem1")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Show OpenCV debug overlay")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--domain-rand", type=int, default=0)
    args = ap.parse_args()

    env = Simulator(
        seed=args.seed,
        map_name=args.map,
        max_steps=args.steps + 1,
        domain_rand=int(bool(args.domain_rand)),
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4,
        full_transparency=True,
        distortion=False,
    )

    obs = env.reset()
    controller = LaneServoController(img_shape=obs.shape)

    step = 0

    try:
        while True:
            action = controller.compute_action(obs)

            """Handle traffic light with state machine"""
            action, light_state = controller.handle_traffic_light(obs, action)

            """Handle stop sign with state machine"""
            action, stop_state = controller.handle_stop_sign(obs, action)

            """Handle slow sign with state machine"""
            action, slow_state = controller.handle_slow_sign(obs, action)

            mask_left, mask_right = detect_lane_markings(obs)

            obs, reward, done, info = env.step(action)
            if not args.no_gui:
                env.render()

            if args.debug:
                dbg = make_debug_frame(obs, mask_left, mask_right)
                cv2.imshow("Lane Servo Debug", dbg)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC quits
                    break

            if done:
                obs = env.reset()
                # Reset state machines when episode resets
                controller.traffic_light_state = "NORMAL"
                controller.stop_sign_state = "NORMAL"
                controller.slow_sign_state = "NORMAL"
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        if args.debug:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()