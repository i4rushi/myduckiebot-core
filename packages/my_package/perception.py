import numpy as np
import cv2

def detect_lane_markings(image_rgb: np.ndarray):
    """
    Segment left (yellow) and right (white) lane markings.
    Returns binary masks normalized to [0,1].
    """
    h, w, _ = image_rgb.shape

    # Region of interest: bottom ~60%
    roi = np.zeros((h, w), np.uint8)
    y0 = int(h * 0.4)
    cv2.rectangle(roi, (0, y0), (w, h), 255, -1)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Yellow mask
    lower_yellow = np.array([15,  80,  80], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # White mask
    lower_white = np.array([0,   0, 190], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask_w = cv2.inRange(hsv, lower_white, upper_white)

    # Apply ROI
    mask_y = cv2.bitwise_and(mask_y, roi)
    mask_w = cv2.bitwise_and(mask_w, roi)

    # Morph cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, k, iterations=1)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, k, iterations=1)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, k, iterations=1)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, k, iterations=1)

    return (mask_y.astype(np.float32) / 255.0,
            mask_w.astype(np.float32) / 255.0)

def detect_stop_line(image_rgb: np.ndarray) -> bool:
    """
    Detect red stop line at the bottom of the image
    """
    h, w, _ = image_rgb.shape
    # Look at bottom 20% of image
    roi_bottom = image_rgb[int(0.8*h):h, :]

    hsv = cv2.cvtColor(roi_bottom, cv2.COLOR_RGB2HSV)

    # Red color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    # Count red pixels
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = roi_bottom.shape[0] * roi_bottom.shape[1]

    # If more than 5% of bottom area is red, consider it a stop line
    return (red_pixels / total_pixels) > 0.05

def detect_light(image):
        h, w, _ = image.shape
        roi = image[0:int(h/5), int(w/3):int(2*w/3)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 120])
        upper_red2 = np.array([179, 255, 255])
        lower_green = np.array([40, 120, 120])
        upper_green = np.array([90, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles( gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=15, minRadius=5, maxRadius=30 )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            h_roi, w_roi = red_mask.shape
            for (x, y, r) in circles[0, :]:
                if 0 <= x < w_roi and 0 <= y < h_roi:
                    if red_mask[y, x] > 0:
                        return "RED"
                    elif green_mask[y,x] > 0:
                        return "GREEN"

            return None

def detect_sign(image):
    h, w, _ = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    """STOP"""
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    """SLOW"""
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 300:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) >= 6:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.putText(image, "STOP", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                return "STOP"

    for cnt in contours_yellow:
        area = cv2.contourArea(cnt)
        if area > 300:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w_box, h_box = cv2.boundingRect(cnt)

                aspect_r = float(w_box) / h_box
                if not (0.8 < aspect_r < 1.2):
                    continue

                rect = cv2.minAreaRect(cnt)
                angle = rect[2]

                if not (20 < abs(angle) < 70):
                    continue

                if y + h_box > 0.8 * h:
                    continue

                cv2.putText(image, "SLOW", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                return "SLOW"
    return None