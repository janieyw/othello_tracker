import cv2
import numpy as np
import math

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

# Set up the video capture device (usually 0 for built-in webcam)
cap = cv2.VideoCapture(0)

# Initialize 2D array to store color information of each grid cell
grid_colors = [['-' for i in range(8)] for j in range(8)]

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the green, black, and white color ranges to detect
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 30, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological operations to fill any gaps in the masks
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the masks
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    green_contours = sorted(green_contours, key=lambda x: cv2.contourArea(x), reverse=True)
    black_contours = sorted(black_contours, key=lambda x: cv2.contourArea(x), reverse=True)
    white_contours = sorted(white_contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Find the largest contour
    largest_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            largest_contour = approx
            break

    intersection_points = []

    # Draw a green outline around the largest contour
    if largest_contour is not None:
        # cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)

        top_left, top_right, bottom_right, bottom_left = largest_contour.reshape(4, 2)

        # cv2.circle(frame, (top_left[0], top_left[1]), 20, (255, 0, 0), -1)  # Top left
        # cv2.circle(frame, (top_right[0], top_right[1]), 20, (255, 0, 0), -1)  # Top Right
        # cv2.circle(frame, (bottom_right[0], bottom_right[1]), 20, (255, 0, 0), -1)  # Bottom Right
        # cv2.circle(frame, (bottom_left[0], bottom_left[1]), 20, (255, 0, 0), -1)  # Bottom Left

        # define the grid size
        num_segments = 8

        # Define the starting points of the quadrilateral
        start_points = [top_left, bottom_left, bottom_right, top_right]

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

        # # Divide each side of the quadrilateral into eight segments to create the grid
        # grid_points = []
        # for i in range(9):
        #     for j in range(9):
        #         # x = top_left[0] + i * (top_side_length / 8) * np.cos(top_side_angle) + j * (
        #         #             left_side_length / 8) * np.cos(left_side_angle)
        #         # y = top_left[1] + i * (top_side_length / 8) * np.sin(top_side_angle) + j * (
        #         #             left_side_length / 8) * np.sin(left_side_angle)
        #         # grid_points.append((x, y))
        #         # x = bottom_left[0] + i * (bottom_side_length / 8) * np.cos(bottom_side_angle) + j * (
        #         #             left_side_length / 8) * np.cos(left_side_angle)
        #         # y = top_left[1] + i * (top_side_length / 8) * np.sin(top_side_angle) + j * (
        #         #             left_side_length / 8) * np.sin(left_side_angle)
        #         # grid_points.append((x, y))
        #         x = bottom_left[0] + i * (bottom_side_length / 8) * np.cos(bottom_side_angle) + j * (
        #                 left_side_length / 8) * np.cos(left_side_angle)
        #         y = top_left[1] + i * (top_side_length / 8) * np.sin(top_side_angle) + j * (
        #                 left_side_length / 8) * np.sin(left_side_angle)
        #         grid_points.append((x, y))
        #
        # # # Calculate the length and angle of each side of the quadrilateral
        # # side_lengths = []
        # # side_angles = []
        # # for i in range(4):
        # #     j = (i + 1) % 4
        # #     dx = corners[j][0] - corners[i][0]
        # #     dy = corners[j][1] - corners[i][1]
        # #     length = math.sqrt(dx * dx + dy * dy)
        # #     angle = math.atan2(dy, dx)
        # #     side_lengths.append(length)
        # #     side_angles.append(angle)
        # #
        # # # Divide each side into eight equal segments
        # # cell_lengths = [l / 8 for l in side_lengths]
        # #
        # # # Initialize the 2D array to store the coordinates of each cell's four corners
        # # cell_corners = np.zeros((8, 8, 4, 2), dtype=np.int32)
        # #
        # # # Calculate the coordinates of each cell's four corners
        # # for i in range(8):
        # #     for j in range(8):
        # #         # Calculate the coordinates of the four intersection points
        # #         x1 = corners[0][0] + (i / 8) * math.cos(side_angles[0]) * side_lengths[0]
        # #         y1 = corners[0][1] + (i / 8) * math.sin(side_angles[0]) * side_lengths[0]
        # #         x2 = corners[1][0] + (j / 8) * math.cos(side_angles[1]) * side_lengths[1]
        # #         y2 = corners[1][1] + (j / 8) * math.sin(side_angles[1]) * side_lengths[1]
        # #         x3 = corners[2][0] + ((7 - i) / 8) * math.cos(side_angles[2]) * side_lengths[2]
        # #         y3 = corners[2][1] + ((7 - i) / 8) * math.sin(side_angles[2]) * side_lengths[2]
        # #         x4 = corners[3][0] + ((7 - j) / 8) * math.cos(side_angles[3]) * side_lengths[3]
        # #         y4 = corners[3][1] + ((7 - j) / 8) * math.sin(side_angles[3]) * side_lengths[3]
        # #         # Store the coordinates in the cell_corners array
        # #         cell_corners[i, j, 0] = [x1, y1]
        # #         cell_corners[i, j, 1] = [x2, y2]
        # #         cell_corners[i, j, 2] = [x3, y3]
        # #         cell_corners[i, j, 3] = [x4, y4]
        # #
        # # # Create an empty list to store the points of each smaller quadrilateral
        # # small_quads = []
        #
        # # # Iterate over the rows and columns
        # # for row in range(9):
        # #     for col in range(9):
        # #         # Define the points of the current smaller quadrilateral
        # #         quad_points = [
        # #             (
        # #                 start_points[0][0] + col * left_side_cell_length * math.cos(
        # #                     top_angle) - row * top_side_cell_length * math.cos(left_angle),
        # #                 start_points[0][1] + col * left_side_cell_length * math.sin(
        # #                     top_angle) - row * top_side_cell_length * math.sin(left_angle)
        # #             )
        # #             # (
        # #             #     start_points[1][0] + col * left_side_cell_length * math.cos(bottom_angle) + (
        # #             #             row + 1) * bottom_side_cell_length * math.cos(left_angle),
        # #             #     start_points[1][1] + col * left_side_cell_length * math.sin(bottom_angle) + (
        # #             #             row + 1) * bottom_side_cell_length * math.sin(left_angle)
        # #             # ),
        # #             # (
        # #             #     start_points[1][0] + (col + 1) * right_side_cell_length * math.cos(bottom_angle) + (
        # #             #             row + 1) * bottom_side_cell_length * math.cos(right_angle),
        # #             #     start_points[1][1] + (col + 1) * right_side_cell_length * math.sin(bottom_angle) + (
        # #             #             row + 1) * bottom_side_cell_length * math.sin(right_angle)
        # #             # ),
        # #             # (
        # #             #     start_points[0][0] + (col + 1) * right_side_cell_length * math.cos(
        # #             #         top_angle) + row * top_side_cell_length * math.cos(right_angle),
        # #             #     start_points[0][1] + (col + 1) * right_side_cell_length * math.sin(
        # #             #         top_angle) + row * top_side_cell_length * math.sin(right_angle)
        # #             # )
        # #         ]
        # #
        # #         # Add the points to the list of smaller quadrilaterals
        # #         small_quads.append(quad_points)
        #
        # for point in grid_points:
        #     x, y = int(point[0]), int(point[1])
        #     cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

        top_divisions = np.linspace(top_left, top_right, num=num_segments + 1, endpoint=True)
        right_divisions = np.linspace(top_right, bottom_right, num=num_segments + 1, endpoint=True)
        bottom_divisions = np.linspace(bottom_right, bottom_left, num=num_segments + 1, endpoint=True)
        left_divisions = np.linspace(bottom_left, top_left, num=num_segments + 1, endpoint=True)

        left_divisions_flipped = np.flip(left_divisions, axis=0)
        top_divisions_flipped = np.flip(top_divisions, axis=0)
        right_divisions_flipped = np.flip(right_divisions, axis=0)
        bottom_divisions_flipped = np.flip(bottom_divisions, axis=0)

        # Draw lines connecting corresponding segments on opposite sides
        for i in range(num_segments):
            for j in range(num_segments):
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
                cv2.line(frame, (int(left_top[0]), int(left_top[1])), (int(right_top[0]), int(right_top[1])), color=(0, 255, 0),
                         thickness=2)
                cv2.line(frame, (int(left_bottom[0]), int(left_bottom[1])), (int(right_bottom[0]), int(right_bottom[1])), color=(0, 255, 0),
                         thickness=2) # Left side
                cv2.line(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_left[0]), int(bottom_left[1])), color=(0, 255, 0),
                         thickness=2)
                cv2.line(frame, (int(top_right[0]), int(top_right[1])), (int(bottom_right[0]), int(bottom_right[1])), color=(0, 255, 0),
                         thickness=2)

                # Add outer points of the quadrilateral
                intersection_points.append((int(top_left[0]), int(top_left[1])))
                intersection_points.append((int(top_right[0]), int(top_right[1])))
                intersection_points.append((int(bottom_left[0]), int(bottom_left[1])))
                intersection_points.append((int(bottom_right[0]), int(bottom_right[1])))
                intersection_points.append((int(left_top[0]), int(left_top[1])))
                intersection_points.append((int(left_bottom[0]), int(left_bottom[1])))
                intersection_points.append((int(right_top[0]), int(right_top[1])))
                intersection_points.append((int(right_bottom[0]), int(right_bottom[1])))

                # Draw lines connecting opposite sides
                line1 = (int(left_top[0]), int(left_top[1]), int(right_top[0]), int(right_top[1]))
                line2 = (int(left_bottom[0]), int(left_bottom[1]), int(right_bottom[0]), int(right_bottom[1]))
                line3 = (int(top_left[0]), int(top_left[1]), int(bottom_left[0]), int(bottom_left[1]))
                line4 = (int(top_right[0]), int(top_right[1]), int(bottom_right[0]), int(bottom_right[1]))

                # Compute intersection points
                for line in [line1, line2, line3, line4]:
                    for other_line in [line1, line2, line3, line4]:
                        if line != other_line:
                            intersection_point = compute_intersection(line, other_line)
                            if intersection_point is not None:
                                intersection_points.append(intersection_point)

        # Display intersection points
        for point in intersection_points:
            cv2.circle(frame, point, radius=20, color=(0, 0, 255), thickness=-1)

        # roi_colors = []
        #
        # for i in range(len(left_divisions) - 1):
        #     for j in range(len(top_divisions) - 1):
        #         x1, y1 = int(j * frame.shape[1] / 8), int(i * frame.shape[0] / 8)
        #         x2, y2 = int((j + 1) * frame.shape[1] / 8), int((i + 1) * frame.shape[0] / 8)
        #
        #         # Apply the color masks to the original image
        #         green_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=green_mask[y1:y2, x1:x2])
        #         black_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=black_mask[y1:y2, x1:x2])
        #         white_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=white_mask[y1:y2, x1:x2])
        #
        #         # Calculate the percentage of each color in the cell
        #         green_percentage = np.count_nonzero(green_pixels) / ((x2 - x1) * (y2 - y1))
        #         black_percentage = np.count_nonzero(black_pixels) / ((x2 - x1) * (y2 - y1))
        #         white_percentage = np.count_nonzero(white_pixels) / ((x2 - x1) * (y2 - y1))
        #
        #         # Find the color with the highest percentage
        #         max_percentage = max(green_percentage, white_percentage, black_percentage)
        #         if green_percentage == max_percentage:
        #             grid_colors[i][j] = '-'
        #         elif black_percentage == max_percentage:
        #             grid_colors[i][j] = 'X'
        #         else:
        #             grid_colors[i][j] = 'O'
        #
        # # Print out the color information for each grid cell
        # if cv2.waitKey(1) & 0xFF == ord(' '):
        #     for row in grid_colors:
        #         print(' '.join(row))
        #     print("---------------")

    # Display the resulting frame
    cv2.imshow('Othello Tracker', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
