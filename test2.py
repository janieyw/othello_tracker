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

# Initialize 2D array to store color information of each grid cell
grid_colors = [['-' for i in range(8)] for j in range(8)]

# Define a variable to keep track of the last time the intersection points were reset
last_reset_time = time.time()

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

        # intersection_points.append((top_left[0], top_left[1]))
        # intersection_points.append((top_right[0], top_right[1]))
        # intersection_points.append((bottom_right[0], bottom_right[1]))
        # intersection_points.append((bottom_left[0], bottom_left[1]))

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
                # cv2.line(frame, (int(left_top[0]), int(left_top[1])), (int(right_top[0]), int(right_top[1])), color=(0, 255, 0),
                #          thickness=2)
                # cv2.line(frame, (int(left_bottom[0]), int(left_bottom[1])), (int(right_bottom[0]), int(right_bottom[1])), color=(0, 255, 0),
                #          thickness=2) # Left side
                # cv2.line(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_left[0]), int(bottom_left[1])), color=(0, 255, 0),
                #          thickness=2)
                # cv2.line(frame, (int(top_right[0]), int(top_right[1])), (int(bottom_right[0]), int(bottom_right[1])), color=(0, 255, 0),
                #          thickness=2)

                # Add outer points of the quadrilateral if they haven't been added yet
                outer_points = [(int(top_left[0]), int(top_left[1])),
                                 (int(top_right[0]), int(top_right[1])),
                                 (int(bottom_left[0]), int(bottom_left[1])),
                                 (int(bottom_right[0]), int(bottom_right[1])),
                                 (int(left_top[0]), int(left_top[1])),
                                 (int(left_bottom[0]), int(left_bottom[1])),
                                 (int(right_top[0]), int(right_top[1])),
                                 (int(right_bottom[0]), int(right_bottom[1]))]

                for point in outer_points:
                    if point not in intersection_points:
                        intersection_points.append(point)

                # Draw lines connecting opposite sides
                line1 = (int(left_top[0]), int(left_top[1]), int(right_top[0]), int(right_top[1]))
                line2 = (int(left_bottom[0]), int(left_bottom[1]), int(right_bottom[0]), int(right_bottom[1]))
                line3 = (int(top_left[0]), int(top_left[1]), int(bottom_left[0]), int(bottom_left[1]))
                line4 = (int(top_right[0]), int(top_right[1]), int(bottom_right[0]), int(bottom_right[1]))

                # Compute intersection points
                for line in [line1, line2, line3, line4]:
                    for other_line in [line1, line2, line3, line4]:
                        if line != other_line:
                            point = compute_intersection(line, other_line)
                            if point is not None and point not in intersection_points and point not in outer_points:
                                intersection_points.append(point)

        # Display intersection points
        for point in intersection_points:
            cv2.circle(frame, point, radius=10, color=(0, 0, 255), thickness=-1)

        print(len(intersection_points))

    # Reset intersection_points every 2 seconds
    if time.time() - last_reset_time > 2:
        intersection_points = []
        last_reset_time = time.time()

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
