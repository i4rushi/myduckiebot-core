#!/usr/bin/env python3
# lane_keeping_pid_turnaware.py
#
# PID-only lane keeping with TURN AWARENESS (stronger turning):
# - Robust vision (CLAHE + LAB+HSV white with right gating; adaptive yellow).
# - Targets lane CENTER when both lines visible.
# - If ONE line disappears for a few frames:
#     * Enter turn mode (left if white disappears; right if yellow disappears).
#     * Hug the remaining line with an inward bias and an explicit steer bias.
#     * Hold the mode for a fixed number of frames, then re-evaluate.
#     * Stronger default bias + speed reduction + short ramp for harder turns.
# - If BOTH disappear: recovery (slow + steering sweep) to re-acquire.
#
# Examples:
#   python lane_keeping_pid_turnaware.py --env-name small_loop --use-white-fallback --base-speed 0.30
#   python lane_keeping_pid_turnaware.py --show-masks --white-right-gate 0.60
#
# Controls:
#   ESC to exit.

import argparse
import time
from dataclasses import dataclass
import numpy as np
import cv2
from gym_duckietown.envs import DuckietownEnv


# =============================== Configs =================================== #

@dataclass
class PIDGains:
    kp: float = 0.010
    ki: float = 0.0003
    kd: float = 0.003
    integral_clamp: float = 5000.0


@dataclass
class SpeedCfg:
    base: float = 0.35
    min_: float = 0.08
    max_: float = 0.55


@dataclass
class VisionCfg:
    roi_top: float = 0.35           # analyze lower 65% of the image
    slices: int = 6
    center_ema_alpha: float = 0.35  # smoothing on detected lane center (px)
    steer_blend: float = 0.70       # low-pass on steering output
    deadband_px: int = 8
    min_correction: float = 0.03
    max_steer: float = 0.75
    white_right_gate: float = 0.58  # fraction of width (search white to right of this)


@dataclass
class RecoveryCfg:
    lost_frames_to_search: int = 10  # frames before entering active search
    sweep_gain: float = 0.55         # steer amplitude during search
    sweep_period_frames: int = 12    # ~0.6s at 20 FPS


@dataclass
class TurnCfg:
    # Enter turn mode if one side is missing this many consecutive frames
    disappear_hysteresis: int = 3
    # Hold turn mode at least this many frames once engaged
    hold_frames: int = 30
    # How strongly to bias steering during turn mode (added to PID)
    steer_bias: float = 0.40
    # When only one line is visible, estimate lane center at
    #   x_line +/- (lane_width_ema * single_line_center_factor)
    single_line_center_factor: float = 0.30  # < 0.50 hugs inside of turn
    # Extra speed reduction factor while in turn mode (multiply base speed)
    speed_scale: float = 0.70
    # Optional ramp factor: up to +ramp_gain at the beginning of turn mode
    ramp_gain: float = 0.40  # 0.40 => up to +40% stronger at the start


# =============================== Controller ================================= #

class LaneKeeperPIDTurnAware:
    """
    PID lane keeping centered on lane, with turn-awareness and robust vision.
    """
    def __init__(
        self,
        pid: PIDGains,
        speed: SpeedCfg,
        vis: VisionCfg,
        recov: RecoveryCfg,
        turn: TurnCfg,
        use_white_fallback: bool = True,
        show_masks: bool = False,
    ):
        self.pid = pid
        self.speed = speed
        self.vis = vis
        self.recov = recov
        self.turn = turn
        self.use_white_fallback = use_white_fallback
        self.show_masks = show_masks

        # state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_steer = 0.0
        self.last_center_ema = None
        self.last_lane_width_ema = None
        self.last_line_side = 0
        self.lost_counter = 0
        self.search_phase = 1
        self.search_phase_frames = 0

        # turn awareness state
        self.turn_mode = None            # None | "left" | "right"
        self.turn_hold = 0               # frames left to hold current turn mode
        self.miss_yellow_frames = 0
        self.miss_white_frames = 0

    # ------------------------------ Vision -------------------------------- #

    @staticmethod
    def _preprocess(rgb):
        """CLAHE on LAB L; return HSV and LAB from enhanced image."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        Lc = clahe.apply(L)
        lab_enh = cv2.merge([Lc, A, B])
        bgr_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr_enh, cv2.COLOR_BGR2LAB)
        return hsv, lab

    @staticmethod
    def _adaptive_yellow_mask(hsv):
        """Brightness-aware yellow mask (HSV)."""
        v = hsv[..., 2]
        v_med = float(np.median(v))
        s_lo = max(60, int(0.9 * (80 - (v_med - 128) * 0.15)))
        v_lo = max(60, int(0.9 * (80 - (v_med - 128) * 0.10)))
        lower = np.array([15, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([40, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def _white_mask(self, hsv, lab, right_gate_x=None, y1=None, y2=None):
        """
        LAB+HSV + adaptive-threshold white mask restricted to lower ROI and right half.
        """
        h, w = hsv.shape[:2]
        L, A, B = cv2.split(lab)

        # HSV gate: bright & low saturation (loosen V to 185 if needed)
        hsv_gate = cv2.inRange(hsv, (0, 0, 200), (180, 80, 255))

        # LAB gates: near-neutral chroma + bright L
        cA = np.full_like(A, 128, dtype=A.dtype)
        cB = np.full_like(B, 128, dtype=B.dtype)
        a_dev = cv2.absdiff(A, cA)
        b_dev = cv2.absdiff(B, cB)
        chroma_ok = cv2.threshold(a_dev + b_dev, 25, 255, cv2.THRESH_BINARY_INV)[1]
        L_hi = cv2.threshold(L, 190, 255, cv2.THRESH_BINARY)[1]

        # Adaptive L (handles uneven lighting)
        L_adapt = cv2.adaptiveThreshold(
            L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -7
        )

        white = cv2.bitwise_and(hsv_gate, L_hi)
        white = cv2.bitwise_or(white, cv2.bitwise_and(chroma_ok, L_adapt))

        # Lower-ROI
        if y1 is None: y1 = int(h * self.vis.roi_top)
        if y2 is None: y2 = h
        vmask = np.zeros_like(white); vmask[y1:y2, :] = 255
        white = cv2.bitwise_and(white, vmask)

        # Right-half gating
        if right_gate_x is None:
            right_gate_x = int(w * self.vis.white_right_gate)
        rmask = np.zeros_like(white); rmask[:, right_gate_x:] = 255
        white = cv2.bitwise_and(white, rmask)

        # Morph cleanup
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return white

    def _slice_centers(self, bin_mask, y1, y2, vis=None, color_bgr=(0, 255, 255)):
        """Per-slice Hough to get x-centers. Returns (centers, weights)."""
        h, w = bin_mask.shape
        centers = []
        weights = np.linspace(0.6, 1.0, self.vis.slices)
        for i, weight in enumerate(weights):
            ys = y1 + i * (y2 - y1) // self.vis.slices
            ye = y1 + (i + 1) * (y2 - y1) // self.vis.slices
            roi = bin_mask[ys:ye, :]
            roi = cv2.GaussianBlur(roi, (5, 5), 0)
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            edges = cv2.Canny(roi, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12, minLineLength=6, maxLineGap=6)

            xs = []
            if lines is not None:
                for x1l, y1l, x2l, y2l in lines[:, 0]:
                    dx = (x2l - x1l); dy = (y2l - y1l)
                    slope = 1e6 if abs(dx) < 1 else dy / dx
                    if abs(slope) < 0.25:  # ignore near-horizontal clutter
                        continue
                    xs.append((x1l + x2l) * 0.5)
                    if vis is not None:
                        cv2.line(vis, (x1l, ys + y1l), (x2l, ys + y2l), color_bgr, 1)
            if xs:
                centers.append(weight * float(np.mean(xs)))
        return centers, weights

    @staticmethod
    def _edge_histogram_peak(bin_mask, y1, y2):
        """Rightmost 40% edge peak as a coarse white fallback."""
        h, w = bin_mask.shape
        xs = slice(int(w * 0.60), w)
        roi = bin_mask[y1:y2, xs]
        edges = cv2.Canny(roi, 50, 150)
        hist = edges.sum(axis=0)
        if hist.size == 0 or hist.max() < 200:
            return None
        peak = int(hist.argmax())
        return int(w * 0.60) + peak

    def _update_lane_width_ema(self, yellow_x, white_x):
        width = abs(white_x - yellow_x)
        if self.last_lane_width_ema is None:
            self.last_lane_width_ema = width
        else:
            self.last_lane_width_ema = 0.8 * self.last_lane_width_ema + 0.2 * width

    def _estimate_center_from_single(self, x_line, which, w, factor=None):
        """Estimate lane center given one line using lane-width EMA and factor."""
        if self.last_lane_width_ema is None:
            self.last_lane_width_ema = 0.25 * w  # bootstrap
        f = self.turn.single_line_center_factor if factor is None else factor
        if which == "yellow":  # white is to the right
            return x_line + f * self.last_lane_width_ema
        else:                  # which == "white"
            return x_line - f * self.last_lane_width_ema

    # ------------------------------- Step --------------------------------- #

    def step(self, obs):
        """Compute (steer, speed, debug_vis) for a single observation."""
        hsv, lab = self._preprocess(obs)
        h, w = hsv.shape[:2]
        y1 = int(h * self.vis.roi_top); y2 = h
        vis = obs.copy()

        # Masks
        mask_y = self._adaptive_yellow_mask(hsv)
        mask_w = self._white_mask(hsv, lab, right_gate_x=int(w * self.vis.white_right_gate), y1=y1, y2=y2)

        # Centers
        y_centers, weights = self._slice_centers(mask_y, y1, y2, vis, (0, 255, 255))
        w_centers, _ = self._slice_centers(mask_w, y1, y2, vis, (255, 255, 255))
        if self.use_white_fallback and not w_centers:
            peak_x = self._edge_histogram_peak(mask_w, y1, y2)
            if peak_x is not None:
                w_centers = [float(peak_x)]

        if self.show_masks:
            cv2.imshow("mask_y", mask_y)
            cv2.imshow("mask_w", mask_w)

        got_y = len(y_centers) > 0
        got_w = len(w_centers) > 0

        # Track disappearances with hysteresis
        self.miss_yellow_frames = 0 if got_y else (self.miss_yellow_frames + 1)
        self.miss_white_frames  = 0 if got_w else (self.miss_white_frames  + 1)

        yellow_x = float(np.sum(y_centers) / max(1e-6, np.sum(weights))) if got_y else None
        white_x  = float(np.sum(w_centers) / max(1e-6, np.sum(weights))) if got_w else None

        # Compute lane center target (pre-turn-mode)
        lane_center_x = None
        if got_y and got_w:
            lane_center_x = 0.5 * (yellow_x + white_x)
            self._update_lane_width_ema(yellow_x, white_x)
            self.lost_counter = 0
        elif got_y:
            lane_center_x = self._estimate_center_from_single(yellow_x, "yellow", w, factor=0.50)
            self.lost_counter = 0
        elif got_w:
            lane_center_x = self._estimate_center_from_single(white_x, "white", w, factor=0.50)
            self.lost_counter = 0
        else:
            self.lost_counter += 1

        # Enter/refresh turn mode based on disappearances
        if self.turn_mode is None:
            if self.miss_yellow_frames >= self.turn.disappear_hysteresis and got_w:
                # Yellow gone, white present -> RIGHT turn
                self.turn_mode = "right"
                self.turn_hold = self.turn.hold_frames
            elif self.miss_white_frames >= self.turn.disappear_hysteresis and got_y:
                # White gone, yellow present -> LEFT turn
                self.turn_mode = "left"
                self.turn_hold = self.turn.hold_frames
        else:
            # refresh / count down
            self.turn_hold -= 1
            if self.turn_hold <= 0 or (got_y and got_w):
                # exit turn mode if time elapsed or both lines back
                self.turn_mode = None
                self.turn_hold = 0

        # If in turn mode, bias target & steering toward inside of corner
        steer_bias = 0.0
        if self.turn_mode == "right":
            # only have white reliably; hug it (center closer than half-width)
            if got_w:
                lane_center_x = self._estimate_center_from_single(
                    white_x, "white", w, factor=self.turn.single_line_center_factor)
            # ramp the bias stronger at start of turn, taper as hold decreases
            ramp = 1.0 + self.turn.ramp_gain * (self.turn_hold / max(1, self.turn.hold_frames))
            steer_bias = -self.turn.steer_bias * ramp  # negative steer is right (we invert later for env)
        elif self.turn_mode == "left":
            if got_y:
                lane_center_x = self._estimate_center_from_single(
                    yellow_x, "yellow", w, factor=self.turn.single_line_center_factor)
            ramp = 1.0 + self.turn.ramp_gain * (self.turn_hold / max(1, self.turn.hold_frames))
            steer_bias = +self.turn.steer_bias * ramp

        # curvature factor from slice deltas (whichever side we have)
        curv_factor = 1.0
        seq = y_centers if got_y else (w_centers if got_w else [])
        if len(seq) > 1:
            diffs = np.diff(seq)
            avg_slope_abs = float(np.mean(np.abs(diffs)))
            curv_factor += 5.0 * avg_slope_abs / (w / 2)
            curv_factor = min(curv_factor, 5.0)

        # ------------------------------ Control ---------------------------- #
        in_recovery = self.lost_counter >= self.recov.lost_frames_to_search

        if (lane_center_x is not None) and not in_recovery:
            # EMA smoothing on target center
            if self.last_center_ema is None:
                self.last_center_ema = lane_center_x
            self.last_center_ema = (
                self.vis.center_ema_alpha * lane_center_x +
                (1 - self.vis.center_ema_alpha) * self.last_center_ema
            )

            error_px = self.last_center_ema - (w / 2)
            self.last_line_side = -1 if error_px < 0 else 1

            if abs(error_px) < self.vis.deadband_px:
                steer = np.sign(error_px) * self.vis.min_correction * curv_factor
                self.integral *= 0.95
            else:
                error_norm = error_px / (w / 2)  # normalize to [-1,1]
                self.integral += error_norm
                self.integral = np.clip(self.integral, -self.pid.integral_clamp, self.pid.integral_clamp)
                derivative = (error_norm - self.last_error)
                steer = (self.pid.kp * error_norm +
                         self.pid.ki * self.integral +
                         self.pid.kd * derivative) * curv_factor
                self.last_error = error_norm

            # Apply turn steer bias if any, then clamp
            steer += steer_bias
            steer = float(np.clip(steer, -self.vis.max_steer, self.vis.max_steer))

            # Speed modulation by steer + extra slowdown in turn mode
            speed_cmd = self.speed.base * (1.1 - 0.6 * min(1.0, abs(steer) / self.vis.max_steer))
            if self.turn_mode is not None:
                speed_cmd *= self.turn.speed_scale
            speed_cmd = float(np.clip(speed_cmd, self.speed.min_, self.speed.max_))
        else:
            # Recovery behavior
            if self.lost_counter < self.recov.lost_frames_to_search:
                steer = 0.35 * self.vis.max_steer * (self.last_line_side if self.last_line_side != 0 else 1)
                speed_cmd = max(self.speed.min_ * 0.7, 0.06)
            else:
                self.search_phase_frames += 1
                if self.search_phase_frames >= self.recov.sweep_period_frames:
                    self.search_phase *= -1
                    self.search_phase_frames = 0
                steer = self.search_phase * self.recov.sweep_gain * self.vis.max_steer
                speed_cmd = 0.10
            self.integral *= 0.90

        # Low-pass filter on steer to reduce jitter
        steer = self.vis.steer_blend * self.last_steer + (1 - self.vis.steer_blend) * steer
        self.last_steer = steer

        # ------------------------------ Viz -------------------------------- #
        if lane_center_x is not None:
            x_draw = int(self.last_center_ema if self.last_center_ema is not None else lane_center_x)
            cv2.line(vis, (x_draw, y1), (x_draw, h - 1), (0, 0, 255), 2)

        if got_y:
            cv2.circle(vis, (int(yellow_x), h - 6), 4, (0, 255, 255), -1)
        if got_w:
            cv2.circle(vis, (int(white_x),  h - 6), 4, (255, 255, 255), -1)

        steer_px = int(steer * w / 2)
        cv2.arrowedLine(vis, (w // 2, h - 1), (w // 2 + steer_px, h - 60), (255, 0, 0), 3)
        hud = f"steer={steer:+.3f} speed={speed_cmd:.2f} lost={self.lost_counter}"
        if self.turn_mode:
            hud += f" turn={self.turn_mode}:{self.turn_hold}"
        if self.last_lane_width_ema is not None:
            hud += f" width_ema={self.last_lane_width_ema:.1f}"
        cv2.putText(vis, hud, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 1, cv2.LINE_AA)

        return float(steer), float(speed_cmd), vis


# ================================= Main ==================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="small_loop")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--use-white-fallback", action="store_true",
                    help="Fuse/estimate white when yellow is weak")
    ap.add_argument("--white-right-gate", type=float, default=0.58,
                    help="Fraction of width; search white to the right of this")
    # PID + speed
    ap.add_argument("--kp", type=float, default=PIDGains.kp)
    ap.add_argument("--ki", type=float, default=PIDGains.ki)
    ap.add_argument("--kd", type=float, default=PIDGains.kd)
    ap.add_argument("--base-speed", type=float, default=SpeedCfg.base)
    ap.add_argument("--min-speed",  type=float, default=SpeedCfg.min_)
    ap.add_argument("--max-speed",  type=float, default=SpeedCfg.max_)
    # recovery
    ap.add_argument("--lost-frames-to-search", type=int, default=RecoveryCfg.lost_frames_to_search)
    ap.add_argument("--sweep-gain", type=float, default=RecoveryCfg.sweep_gain)
    ap.add_argument("--sweep-period-frames", type=int, default=RecoveryCfg.sweep_period_frames)
    # turn awareness
    ap.add_argument("--turn-hysteresis", type=int, default=TurnCfg.disappear_hysteresis)
    ap.add_argument("--turn-hold-frames", type=int, default=TurnCfg.hold_frames)
    ap.add_argument("--turn-steer-bias", type=float, default=TurnCfg.steer_bias)
    ap.add_argument("--turn-center-factor", type=float, default=TurnCfg.single_line_center_factor)
    ap.add_argument("--turn-speed-scale", type=float, default=TurnCfg.speed_scale)
    ap.add_argument("--turn-ramp-gain", type=float, default=TurnCfg.ramp_gain)
    # debug
    ap.add_argument("--show-masks", action="store_true")
    ap.add_argument("--domain-rand", action="store_true")
    args = ap.parse_args()

    pid = PIDGains(kp=args.kp, ki=args.ki, kd=args.kd)
    speed = SpeedCfg(base=args.base_speed, min_=args.min_speed, max_=args.max_speed)
    vis = VisionCfg(white_right_gate=args.white_right_gate)
    recov = RecoveryCfg(lost_frames_to_search=args.lost_frames_to_search,
                        sweep_gain=args.sweep_gain,
                        sweep_period_frames=args.sweep_period_frames)
    turn = TurnCfg(disappear_hysteresis=args.turn_hysteresis,
                   hold_frames=args.turn_hold_frames,
                   steer_bias=args.turn_steer_bias,
                   single_line_center_factor=args.turn_center_factor,
                   speed_scale=args.turn_speed_scale,
                   ramp_gain=args.turn_ramp_gain)

    controller = LaneKeeperPIDTurnAware(
        pid, speed, vis, recov, turn,
        use_white_fallback=args.use_white_fallback,
        show_masks=args.show_masks
    )

    env = DuckietownEnv(
        map_name=args.env_name,
        domain_rand=args.domain_rand,
        draw_curve=False,
        draw_bbox=False
    )
    obs = env.reset()

    dt = 1.0 / max(1e-6, args.fps)
    steps = 0

    try:
        while True:
            steer, speed_cmd, vis_img = controller.step(obs)
            action = np.array([speed_cmd, -steer], dtype=np.float32)  # env expects negative steer sign
            obs, reward, done, info = env.step(action)

            if not args.headless:
                env.render()
                cv2.imshow("Lane Keeping (PID, TURN-AWARE+STRONG)", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break

            time.sleep(dt)

            if done:
                print("[INFO] Resetting environment")
                obs = env.reset()
                # reset controller state but keep configs
                controller = LaneKeeperPIDTurnAware(
                    pid, speed, vis, recov, turn,
                    use_white_fallback=args.use_white_fallback,
                    show_masks=args.show_masks
                )
            steps += 1
    finally:
        env.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[INFO] Finished after {steps} steps")


if __name__ == "__main__":
    main()