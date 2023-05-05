import cv2
import numpy as np
import math
import mediapipe as mp
from constants import BLACK, WHITE, GREEN, TOTAL_DISK_NUM, GRID_SIZE
from utils import compute_intersection, display_in_gradient, print_board, print_line_separator, print_p1_score, print_p2_score, print_round_result

cap = cv2.VideoCapture(0)  # Use iPhone as webcam

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the green, black, and white color ranges to detect
    green_mask = cv2.inRange(hsv, (40, 60, 60), (80, 255, 255))  # looser range: (36, 25, 25), (86, 255, 255) tighter range: (40, 60, 60), (80, 255, 255)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))  # looser: (0, 0, 0), (180, 255, 50) tighter: (0, 0, 0), (180, 50, 50)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))  # looser: (0, 0, 150), (180, 30, 255) tighter: (0, 0, 200), (180, 30, 255)

    # Apply morphological operations to fill any gaps in the masks
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Apply the color masks to the entire frame
    green_pixels = cv2.bitwise_and(frame, frame, mask=green_mask)
    black_pixels = cv2.bitwise_and(frame, frame, mask=black_mask)
    white_pixels = cv2.bitwise_and(frame, frame, mask=white_mask)

    # Find the largest contour
    largest_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            largest_contour = approx
            break

    intersection_points = []

    # Initialize a two-dimensional array to store the corner points of each grid cell
    grid_cells = [['-' for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]

    # Draw a green outline around the largest contour
    if largest_contour is not None:
        # cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)

        top_left, top_right, bottom_right, bottom_left = largest_contour.reshape(4, 2)

        # intersection_points.append((top_left[0], top_left[1]))
        # intersection_points.append((top_right[0], top_right[1]))
        # intersection_points.append((bottom_right[0], bottom_right[1]))
        # intersection_points.append((bottom_left[0], bottom_left[1]))

        ver_lines = set()
        hor_lines = set()

        # Calculate the angles of the sides
        top_angle = math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
        bottom_angle = math.atan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0])
        left_angle = math.atan2(top_left[1] - bottom_left[1], top_left[0] - bottom_left[0])
        right_angle = math.atan2(top_right[1] - bottom_right[1], top_right[0] - bottom_right[0])

        # Calculate the length and angle of each side of the quadrilateral
        top_side_length = np.linalg.norm(np.array(top_right) - np.array(top_left))
        top_side_angle = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0])

        right_side_length = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        right_side_angle = np.arctan2(bottom_right[1] - top_right[1], bottom_right[0] - top_right[0])

        bottom_side_length = np.linalg.norm(np.array(bottom_left) - np.array(bottom_right))
        bottom_side_angle = np.arctan2(bottom_left[1] - bottom_right[1], bottom_left[0] - bottom_right[0])

        left_side_length = np.linalg.norm(np.array(top_left) - np.array(bottom_left))
        left_side_angle = np.arctan2(top_left[1] - bottom_left[1], top_left[0] - bottom_left[0])

        top_divisions = np.linspace(top_left, top_right, num=GRID_SIZE + 1, endpoint=True)
        right_divisions = np.linspace(top_right, bottom_right, num=GRID_SIZE + 1, endpoint=True)
        bottom_divisions = np.linspace(bottom_right, bottom_left, num=GRID_SIZE + 1, endpoint=True)
        left_divisions = np.linspace(bottom_left, top_left, num=GRID_SIZE + 1, endpoint=True)

        left_divisions_flipped = np.flip(left_divisions, axis=0)
        top_divisions_flipped = np.flip(top_divisions, axis=0)
        right_divisions_flipped = np.flip(right_divisions, axis=0)
        bottom_divisions_flipped = np.flip(bottom_divisions, axis=0)

        # Draw lines connecting corresponding segments on opposite sides
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                top_left = top_divisions_flipped[i, :]
                top_right = top_divisions_flipped[i + 1, :]
                bottom_left = bottom_divisions[i, :]
                bottom_right = bottom_divisions[i + 1, :]
                left_top = left_divisions_flipped[j, :]
                left_bottom = left_divisions_flipped[j + 1, :]
                right_top = right_divisions[j, :]
                right_bottom = right_divisions[j + 1, :]

                # Draw smaller quadrilateral
                points = np.array([top_left, top_right, bottom_right, bottom_left])

                # # Draw lines connecting opposite sides
                # cv2.line(frame, (int(left_top[0]), int(left_top[1])), (int(right_top[0]), int(right_top[1])), color=(0, 255, 0),
                #          thickness=2)
                # cv2.line(frame, (int(left_bottom[0]), int(left_bottom[1])), (int(right_bottom[0]), int(right_bottom[1])), color=(0, 255, 0),
                #          thickness=2) # Left side
                # cv2.line(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_left[0]), int(bottom_left[1])), color=(0, 255, 0),
                #          thickness=2)
                # cv2.line(frame, (int(top_right[0]), int(top_right[1])), (int(bottom_right[0]), int(bottom_right[1])), color=(0, 255, 0),
                #          thickness=2)

                # Draw lines connecting opposite sides
                line1 = (int(left_top[0]), int(left_top[1]), int(right_top[0]), int(right_top[1]))
                line2 = (int(left_bottom[0]), int(left_bottom[1]), int(right_bottom[0]), int(right_bottom[1]))
                line3 = (int(top_left[0]), int(top_left[1]), int(bottom_left[0]), int(bottom_left[1]))
                line4 = (int(top_right[0]), int(top_right[1]), int(bottom_right[0]), int(bottom_right[1]))

                ver_lines.add(line1)
                ver_lines.add(line2)
                hor_lines.add(line3)
                hor_lines.add(line4)

                # Add outer points of the quadrilateral if they haven't been added yet
                outer_points = [(int(top_left[0]), int(top_left[1])),
                                (int(top_right[0]), int(top_right[1])),
                                (int(bottom_left[0]), int(bottom_left[1])),
                                (int(bottom_right[0]), int(bottom_right[1])),
                                (int(left_top[0]), int(left_top[1])),
                                (int(left_bottom[0]), int(left_bottom[1])),
                                (int(right_top[0]), int(right_top[1])),
                                (int(right_bottom[0]), int(right_bottom[1]))]

                for outer_point in outer_points:
                    if all(np.linalg.norm(np.array(outer_point) - np.array(point)) > 5 for point in
                           intersection_points):
                        intersection_points.append(outer_point)

        for v_line in ver_lines:
            for h_line in hor_lines:
                start_v = (v_line[0], v_line[1])
                end_v = (v_line[2], v_line[3])
                start_h = (h_line[0], h_line[1])
                end_h = (h_line[2], h_line[3])
                # cv2.line(frame, start_v, end_v, color=(0, 255, 0), thickness=2)
                # cv2.line(frame, start_h, end_h, color=(0, 255, 0), thickness=2)
                intersection_point = compute_intersection(v_line, h_line)
                if intersection_point is not None and all(
                        np.linalg.norm(np.array(intersection_point) - np.array(point)) > 5 for point in
                        intersection_points):
                    intersection_points.append(intersection_point)

        intersection_points = sorted(intersection_points, key=lambda p: (p[1], p[0]))

        # Display intersection points in gradient, while incrementing blue value and decrementing red value
        display_in_gradient(frame, intersection_points, 0, 255)

        # Initialize grid_colors with all '-'
        grid_colors = [[GREEN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        if len(intersection_points) == 81:
            # Loop through each row and column to add the four corner points for each cell
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    top_left = intersection_points[i * 9 + j]
                    top_right = intersection_points[i * 9 + j + 1]
                    bottom_left = intersection_points[(i + 1) * 9 + j]
                    bottom_right = intersection_points[(i + 1) * 9 + j + 1]
                    grid_cells[i][j] = [top_left, top_right, bottom_left, bottom_right]

                    # Draw the quadrilateral
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                    # Define the masks for the current cell rectangle
                    cell_rect = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cell_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(cell_mask, [np.array([top_left, top_right, bottom_right, bottom_left])], 0,
                                     (255, 255, 255), -1)
                    cv2.drawContours(cell_rect, [np.array([top_left, top_right, bottom_right, bottom_left])], 0,
                                     (255, 255, 255), -1)
                    green_mask_cell = cv2.bitwise_and(green_mask, cell_mask)
                    black_mask_cell = cv2.bitwise_and(black_mask, cell_mask)
                    white_mask_cell = cv2.bitwise_and(white_mask, cell_mask)

                    # Apply the color masks to the rectangle
                    green_pixels = cv2.bitwise_and(frame, frame, mask=green_mask_cell)
                    black_pixels = cv2.bitwise_and(frame, frame, mask=black_mask_cell)
                    white_pixels = cv2.bitwise_and(frame, frame, mask=white_mask_cell)

                    # Count the number of pixels of each color
                    green_count = np.count_nonzero(green_pixels) / 81
                    black_count = np.count_nonzero(black_pixels)
                    white_count = np.count_nonzero(white_pixels)

                    # Determine the dominant color
                    if black_count > green_count and black_count > white_count:
                        grid_colors[i][j] = BLACK
                        # Draw a filled circle in the center of the cell
                        center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
                        cv2.circle(frame, center, 20, (0, 0, 0), -1)
                    elif white_count > green_count and white_count > black_count:
                        grid_colors[i][j] = WHITE
                        # Draw an empty circle in the center of the cell
                        center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
                        cv2.circle(frame, center, 20, (255, 255, 255), 2)
                    else:
                        grid_colors[i][j] = GREEN

        # Initialize p1_disk_num and p2_disk_num to 0
        p1_disk_num = 0
        p2_disk_num = 0

        # Loop through the grid_colors array
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid_colors[i][j] == BLACK:
                    p1_disk_num += 1
                elif grid_colors[i][j] == WHITE:
                    p2_disk_num += 1

        # Print out the color information for each grid cell
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print_board(grid_colors)
            print_line_separator()
            print_p1_score(p1_disk_num)
            print_p2_score(p2_disk_num)
            print_round_result(p1_disk_num, p2_disk_num)

    # Display the resulting frame
    cv2.imshow('Othello Tracker', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
