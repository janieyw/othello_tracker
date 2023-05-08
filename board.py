import cv2
import numpy as np
from constants import GREEN, BLACK, WHITE, GRID_SIZE

def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if den == 0:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
    if ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (int(x), int(y))
    else:
        return None

def get_green_mask(hsv, kernel):
    # Define the green, black, and white color ranges to detect
    green_mask = cv2.inRange(hsv, (40, 60, 60),
                             (80, 255, 255))  # looser: (36, 25, 25), (86, 255, 255) tighter: (40, 60, 60), (80, 255, 255)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    return green_mask

def get_black_mask(hsv, kernel):
    # Define the green, black, and white color ranges to detect
    black_mask = cv2.inRange(hsv, (0, 0, 0),
                             (180, 255, 50))  # looser: (0, 0, 0), (180, 255, 50) tighter: (0, 0, 0), (180, 50, 50)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    return black_mask

def get_white_mask(hsv, kernel):
    # Define the green, black, and white color ranges to detect
    white_mask = cv2.inRange(hsv, (0, 0, 200),
                             (180, 30, 255))  # looser: (0, 0, 150), (180, 30, 255) tighter: (0, 0, 200), (180, 30, 255)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    return white_mask

def get_green_mask(hsv, kernel):
    # Define the green, black, and white color ranges to detect
    green_mask = cv2.inRange(hsv, (40, 60, 60),
                             (80, 255, 255))  # looser: (36, 25, 25), (86, 255, 255) tighter: (40, 60, 60), (80, 255, 255)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    return green_mask

def get_color_masks(hsv, kernel):
    green_mask = get_green_mask(hsv, kernel)
    black_mask = get_black_mask(hsv, kernel)
    white_mask = get_white_mask(hsv, kernel)
    return green_mask, black_mask, white_mask

def find_largest_contour(contours):
    largest_contour = None
    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            largest_contour = approx
            break
    return largest_contour

def display_in_gradient(frame, intersection_points, blue_value, red_value):
    for p in range(len(intersection_points)):
        if p == 0:
            radius = 20
        else:
            radius = 8

        # Draw the point with the new color
        cv2.circle(frame, intersection_points[p], radius=radius, color=(blue_value, 0, red_value), thickness=-1)

        # Increment and decrement color value by a fixed amount
        blue_value += 3
        red_value -= 3

def display_grid_lines(frame, start_v, end_v, start_h, end_h):
    cv2.line(frame, start_v, end_v, color=(0, 255, 0), thickness=2)
    cv2.line(frame, start_h, end_h, color=(0, 255, 0), thickness=2)

def reset_player_disk_num():
    return 0, 0

def display_player_num(frame, player_num):
    cv2.putText(frame, f"Player {player_num}", (25, 65), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

def draw_quadrilateral(frame, top_left, bottom_right):
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

def define_corner_points(intersection_points, i, j):
    top_left = intersection_points[i * 9 + j]
    top_right = intersection_points[i * 9 + j + 1]
    bottom_left = intersection_points[(i + 1) * 9 + j]
    bottom_right = intersection_points[(i + 1) * 9 + j + 1]
    return top_left, top_right, bottom_left, bottom_right

def determine_dominant_color(frame, cell_mask, green_mask, black_mask, white_mask):
    # Apply the masks to the color masks
    green_mask_cell = cv2.bitwise_and(green_mask, cell_mask)
    black_mask_cell = cv2.bitwise_and(black_mask, cell_mask)
    white_mask_cell = cv2.bitwise_and(white_mask, cell_mask)

    # Apply the color masks to the frame
    green_pixels = cv2.bitwise_and(frame, frame, mask=green_mask_cell)
    black_pixels = cv2.bitwise_and(frame, frame, mask=black_mask_cell)
    white_pixels = cv2.bitwise_and(frame, frame, mask=white_mask_cell)

    # Count the number of pixels of each color
    green_count = np.count_nonzero(green_pixels) / 81
    black_count = np.count_nonzero(black_pixels)
    white_count = np.count_nonzero(white_pixels)

    # Determine the dominant color
    if black_count > green_count and black_count > white_count:
        return BLACK
    elif white_count > green_count and white_count > black_count:
        return WHITE
    else:
        return GREEN

def draw_disk(frame, color, top_left, bottom_right):
    # Draw a filled or empty circle in the center of the cell
    center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
    if color == BLACK:
        cv2.circle(frame, center, 20, (0, 0, 0), -1)
    elif color == WHITE:
        cv2.circle(frame, center, 20, (255, 255, 255), 2)

def calculate_side_properties(point1, point2):
    length = np.linalg.norm(np.array(point2) - np.array(point1))
    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    return length, angle

def find_intersection_points(top_divisions_flipped, bottom_divisions, left_divisions_flipped, right_divisions):
    intersection_points = []
    ver_lines = set()
    hor_lines = set()

    # Draw lines connecting corresponding segments on opposite sides
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            top_left, top_right = top_divisions_flipped[i:i + 2, :]
            bottom_left, bottom_right = bottom_divisions[i:i + 2, :]
            left_top, left_bottom = left_divisions_flipped[j:j + 2, :]
            right_top, right_bottom = right_divisions[j:j + 2, :]

            # Draw lines connecting opposite sides
            ver_line1 = (int(left_top[0]), int(left_top[1]), int(right_top[0]), int(right_top[1]))
            ver_line2 = (int(left_bottom[0]), int(left_bottom[1]), int(right_bottom[0]), int(right_bottom[1]))
            hor_line1 = (int(top_left[0]), int(top_left[1]), int(bottom_left[0]), int(bottom_left[1]))
            hor_line2 = (int(top_right[0]), int(top_right[1]), int(bottom_right[0]), int(bottom_right[1]))

            ver_lines.update([ver_line1, ver_line2])
            hor_lines.update([hor_line1, hor_line2])

            # Add outer points of the quadrilateral if they haven't been added yet
            outer_points = [(int(top_left[0]), int(top_left[1])), (int(top_right[0]), int(top_right[1])),
                            (int(bottom_left[0]), int(bottom_left[1])), (int(bottom_right[0]), int(bottom_right[1])),
                            (int(left_top[0]), int(left_top[1])), (int(left_bottom[0]), int(left_bottom[1])),
                            (int(right_top[0]), int(right_top[1])), (int(right_bottom[0]), int(right_bottom[1]))]

            for outer_point in outer_points:
                if all(np.linalg.norm(np.array(outer_point) - np.array(point)) > 5 for point in intersection_points):
                    intersection_points.append(outer_point)

    for v_line in ver_lines:
        for h_line in hor_lines:
            # cv2.line(frame, (v_line[0], v_line[1]), (v_line[2], v_line[3]), color=(0, 255, 0), thickness=2)
            # cv2.line(frame, (h_line[0], h_line[1]), (h_line[2], h_line[3]), color=(0, 255, 0), thickness=2)
            intersection_point = compute_intersection(v_line, h_line)
            if intersection_point is not None and all(
                    np.linalg.norm(np.array(intersection_point) - np.array(point)) > 5 for point in
                    intersection_points):
                intersection_points.append(intersection_point)
    return intersection_points

def divide_side_into_segments(point1, point2, num_segments):
    return np.linspace(point1, point2, num=num_segments + 1, endpoint=True)

def flip_divisions(divisions):
    return np.flip(divisions, axis=0)

def get_frame_center(frame):
    height, width = frame.shape[:2]
    center_x, center_y = int(width/2), int(height/2)
    return (center_x, center_y)