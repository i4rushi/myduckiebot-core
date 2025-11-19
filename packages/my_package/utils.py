import numpy as np

def _xy_weights(shape):
    h, w = shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    return x, y ** 1.5

def get_steer_matrix_left_lane_markings(shape):
    x, y = _xy_weights(shape)
    return (-x) * y

def get_steer_matrix_right_lane_markings(shape):
    x, y = _xy_weights(shape)
    return (x) * y

def make_debug_frame(obs_rgb, mask_left, mask_right):
    """Stack RGB, yellow mask, and white mask side by side for debugging."""
    h, w, _ = obs_rgb.shape
    rgb_bgr = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR)
    # Convert masks to 3-channel color images
    mask_y_vis = cv2.applyColorMap((mask_left * 255).astype(np.uint8), cv2.COLORMAP_AUTUMN)
    mask_w_vis = cv2.applyColorMap((mask_right * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    return np.hstack([rgb_bgr, mask_y_vis, mask_w_vis])