import cv2
import numpy as np

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

    # Draw a green outline around the largest contour
    if largest_contour is not None:
        # cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)

        top_left, top_right, bottom_right, bottom_left = largest_contour.reshape(4, 2)

        # # Divide each side of the board contour into 8 sections
        # top_divisions = np.linspace(top_left, top_right, num=9, endpoint=True)
        # right_divisions = np.linspace(top_right, bottom_right, num=9, endpoint=True)
        # bottom_divisions = np.linspace(bottom_right, bottom_left, num=9, endpoint=True)
        # left_divisions = np.linspace(top_left, bottom_left, num=9, endpoint=True)
        #
        # # Connect the dividing points of each side to those of the facing side in the desired order
        # left_divisions_flipped = np.flip(left_divisions, axis=0)
        # top_divisions_flipped = np.flip(top_divisions, axis=0)
        # right_divisions_flipped = np.flip(right_divisions, axis=0)
        # bottom_divisions_flipped = np.flip(bottom_divisions, axis=0)

        cv2.circle(frame, (top_left[0], top_left[1]), 10, (255, 0, 0), -1)
        # cv2.circle(frame, (top_right[0], top_right[1]), 10, (255, 0, 0), -1)
        # cv2.circle(frame, (bottom_right[0], bottom_right[1]), 10, (255, 0, 0), -1)
        # cv2.circle(frame, (bottom_left[0], bottom_left[1]), 10, (255, 0, 0), -1)
        # cv2.line(frame, (top_left[0], top_left[1]), (top_right[0], top_right[1]), (0, 255, 0), 2)
        # cv2.line(frame, (top_right[0], top_right[1]), (bottom_right[0], bottom_right[1]), (0, 255, 0), 2)
        # cv2.line(frame, (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1]), (0, 255, 0), 2)
        # cv2.line(frame, (bottom_left[0], bottom_left[1]), (top_left[0], top_left[1]), (0, 255, 0), 2)
        num_segments = 8

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

                # Draw lines connecting opposite sides
                cv2.line(frame, (int(left_top[0]), int(left_top[1])), (int(right_top[0]), int(right_top[1])), color=(0, 255, 0),
                         thickness=2)
                cv2.line(frame, (int(left_bottom[0]), int(left_bottom[1])), (int(right_bottom[0]), int(right_bottom[1])), color=(0, 255, 0),
                         thickness=2) # Left side
                cv2.line(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_left[0]), int(bottom_left[1])), color=(0, 255, 0),
                         thickness=2)
                cv2.line(frame, (int(top_right[0]), int(top_right[1])), (int(bottom_right[0]), int(bottom_right[1])), color=(0, 255, 0),
                         thickness=2)

        # for i in range(len(left_divisions) - 1):
        #     cv2.line(frame, tuple(map(int, top_divisions_flipped[i])), tuple(map(int, bottom_divisions[i])),
        #              (0, 255, 0), 5)
        #     cv2.line(frame, tuple(map(int, left_divisions[i])), tuple(map(int, right_divisions[i])), (0, 255, 0), 5)
    #         for j in range(len(top_divisions) - 1):
    #             # Apply the color masks to the original image
    #             green_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=green_mask[y1:y2, x1:x2])
    #             black_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=black_mask[y1:y2, x1:x2])
    #             white_pixels = cv2.bitwise_and(frame[y1:y2, x1:x2], frame[y1:y2, x1:x2], mask=white_mask[y1:y2, x1:x2])
    #
    #             # Calculate the percentage of each color in the cell
    #             green_percentage = np.count_nonzero(green_pixels) / ((x2 - x1) * (y2 - y1))
    #             black_percentage = np.count_nonzero(black_pixels) / ((x2 - x1) * (y2 - y1))
    #             white_percentage = np.count_nonzero(white_pixels) / ((x2 - x1) * (y2 - y1))
    #
    #             # Find the color with the highest percentage
    #             max_percentage = max(green_percentage, white_percentage, black_percentage)
    #             if green_percentage == max_percentage:
    #                 grid_colors[i][j] = 'G'
    #             elif black_percentage == max_percentage:
    #                 grid_colors[i][j] = 'B'
    #             else:
    #                 grid_colors[i][j] = 'W'
    #
    #             x1, y1 = int(top_divisions_flipped[i][0]), int(left_divisions[j][1])
    #             x2, y2 = x2 - int(bottom_divisions[i][0]), y2 + int(right_divisions[j][1])
    #
    #
    # if cv2.waitKey(1) & 0xFF == ord(' '):
    #     for row in grid_colors:
    #         print(' '.join(row))
    #     print("---------------")

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
