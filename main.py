import cv2
import numpy as np
import math
import time

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

while True:

    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the green, black, and white color ranges to detect
    green_mask = cv2.inRange(hsv, (40, 60, 60), (80, 255, 255))  # looser range: (36, 25, 25), (86, 255, 255) tighter range: (40, 60, 60), (80, 255, 255)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
    white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 30, 255))

    # Apply morphological operations to fill any gaps in the masks
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

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

    # define the grid size
    GRID_SIZE = 8

    # Initialize a two-dimensional array to store the corner points of each grid cell
    grid_cells = [['-' for i in range(8)] for j in range(8)]

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

        # Define the starting blue value
        blue_value = 0
        red_value = 255

        # Display intersection points with increasing blue color
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

        grid_colors = [[0 for x in range(8)] for y in range(8)]
        if len(intersection_points) == 81:
            # Loop through each row and column to add the four corner points for each cell
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    top_left = intersection_points[i * 9 + j]
                    top_right = intersection_points[i * 9 + j + 1]
                    bottom_left = intersection_points[(i + 1) * 9 + j]
                    bottom_right = intersection_points[(i + 1) * 9 + j + 1]
                    grid_cells[i][j] = [top_left, top_right, bottom_left, bottom_right]

            # Loop through each grid cell and draw its quadrilateral
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    # Get the four corner points of the grid cell
                    top_left, top_right, bottom_left, bottom_right = grid_cells[i][j]

                    # Draw the quadrilateral
                    pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), thickness=2)

            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    # Get the four corner points of the grid cell
                    top_left, top_right, bottom_left, bottom_right = grid_cells[i][j]

                    x1 = top_left[0]
                    y1 = top_left[1]
                    x2 = bottom_right[0]
                    y2 = bottom_right[1]

                    # Apply the color masks to the original image
                    green_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=green_mask[y1:y2, x1:x2])
                    black_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=black_mask[y1:y2, x1:x2])
                    white_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=white_mask[y1:y2, x1:x2])

                    area = (x2 - x1) * (y2 - y1)
                    if area == 0:
                        green_percentage = 0
                        black_percentage = 0
                        white_percentage = 0
                    else:
                        green_percentage = np.count_nonzero(green_pixels) / area
                        black_percentage = np.count_nonzero(black_pixels) / area
                        white_percentage = np.count_nonzero(white_pixels) / area

                    # Find the color with the highest percentage
                    max_percentage = max(green_percentage, white_percentage, black_percentage)
                    if black_percentage == max_percentage:
                        grid_colors[i][j] = '0'
                    elif white_percentage == max_percentage:
                        grid_colors[i][j] = '1'
                    else:
                        grid_colors[i][j] = '-'

        # Print out the color information for each grid cell every five seconds
        if int(time.time()) % 5 == 0:
            for row in grid_colors:
                print(' '.join(str(elem) for elem in row))
            print("---------------")

        # Print out the color information for each grid cell
        if cv2.waitKey(1) & 0xFF == ord(' '):
            for row in grid_colors:
                print(' '.join(str(elem) for elem in row))  # print(' '.join(row))
            print("---------------")

    # Display the resulting frame
    cv2.imshow('Othello Tracker', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
