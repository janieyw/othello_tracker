# import cv2
# import numpy as np
#
# # Set up the video capture device (usually 0 for built-in webcam)
# cap = cv2.VideoCapture(0)
#
# while True:
#     # Read a frame from the video capture device
#     ret, frame = cap.read()
#
#     # Convert the frame to the HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Define the green color range to detect
#     lower_green = np.array([40, 50, 50])
#     upper_green = np.array([80, 255, 255])
#
#     # Create a mask based on the green color range
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#
#     # Apply a morphological operation to fill any gaps in the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Iterate over the contours and filter for ones that are roughly rectangular and have a large area
#     board_contour = None
#     max_area = 0
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
#         if len(approx) == 4 and area > max_area:
#             board_contour = approx
#             max_area = area
#
#     # Draw a green outline around the detected board
#     if board_contour is not None:
#         cv2.drawContours(frame, [board_contour], 0, (0, 255, 0), 2)
#
#         # Compute the average x coordinate of each vertical line
#         x_coords = [point[0][0] for point in board_contour]
#         x_coords = sorted(x_coords)
#         x_coords_avg = [sum(x_coords[i:i + 8]) / 8 for i in range(0, len(x_coords), 8)]
#
#         # Compute the average y coordinate of each horizontal line
#         y_coords = [point[0][1] for point in board_contour]
#         y_coords = sorted(y_coords)
#         y_coords_avg = [sum(y_coords[i:i + 8]) / 8 for i in range(0, len(y_coords), 8)]
#
#         # Compute the distance between the two farthest vertical lines
#         vert_dist = max(x_coords) - min(x_coords)
#
#         # Compute the distance between the two farthest horizontal lines
#         horiz_dist = max(y_coords) - min(y_coords)
#
#         # Compute the horizontal and vertical spacing between the cells
#         cell_width = vert_dist / 8
#         cell_height = horiz_dist / 8
#
#         # Create an approximate map of the cell centers' (x, y) values
#         cell_centers = []
#         for i in range(8):
#             if i < len(y_coords_avg):  # check if index is valid
#                 for j in range(8):
#                     if j < len(x_coords_avg):
#                         x = int(x_coords_avg[j] + (i + 0.5) * cell_width)
#                         y = int(y_coords_avg[i] + (j + 0.5) * cell_height)
#                         cell_centers.append((x, y))
#
#         # Draw vertical grid lines
#         for i in range(9):
#             x = int(min(x_coords) + i * cell_width)
#             cv2.line(frame, (x, min(y_coords)), (x, max(y_coords)), (0, 0, 255), 1)
#
#         # Draw horizontal grid lines
#         for i in range(9):
#             y = int(min(y_coords) + i * cell_height)
#             cv2.line(frame, (min(x_coords), y), (max(x_coords), y), (0, 0, 255), 1)
#
#     # Display the original frame with the rectangle and dots drawn
#     cv2.imshow('Frame', frame)
#
#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture device and close all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Set up the video capture device (usually 0 for built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the green color range to detect
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Create a mask based on the green color range
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply a morphological operation to fill any gaps in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and filter for ones that are roughly rectangular and have a large area
    board_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and area > max_area:
            board_contour = approx
            max_area = area

    # Draw a green outline around the detected board
    if board_contour is not None:
        cv2.drawContours(frame, [board_contour], 0, (0, 255, 0), 2)

        # Compute the average x coordinate of each vertical line
        x_coords = [point[0][0] for point in board_contour]
        x_coords = sorted(x_coords)
        x_coords_avg = [sum(x_coords[i:i + 8]) / 8 for i in range(0, len(x_coords), 8)]

        # Compute the average y coordinate of each horizontal line
        y_coords = [point[0][1] for point in board_contour]
        y_coords = sorted(y_coords)
        y_coords_avg = [sum(y_coords[i:i + 8]) / 8 for i in range(0, len(y_coords), 8)]

        # Compute the distance between the two farthest vertical lines
        vert_dist = max(x_coords) - min(x_coords)

        # Compute the distance between the two farthest horizontal lines
        horiz_dist = max(y_coords) - min(y_coords)

        # Compute the horizontal and vertical spacing between the cells
        cell_width = vert_dist / 8
        cell_height = horiz_dist / 8

        # Create an approximate map of the cell centers' (x, y) values
        cell_centers = []
        for i in range(8):
            if i < len(y_coords_avg):  # check if index is valid
                for j in range(8):
                    if j < len(x_coords_avg):
                        x = int(x_coords_avg[j] + (i + 0.5) * cell_width)
                        y = int(y_coords_avg[i] + (j + 0.5) * cell_height)
                        cell_centers.append((x, y))

        # Draw vertical grid lines
        for i in range(9):
            x = int(min(x_coords) + i * cell_width)
            cv2.line(frame, (x, min(y_coords)), (x, max(y_coords)), (0, 255, 0), 2)

        # Draw horizontal grid lines
        for i in range(9):
            y = int(min(y_coords) + i * cell_height)
            cv2.line(frame, (min(x_coords), y), (max(x_coords), y), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Wait for a key press and check if it is the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device and close all windows
cap.release()
cv2.destroyAllWindows()
