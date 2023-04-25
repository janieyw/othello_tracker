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

        top_left, top_right, bottom_right, bottom_left = board_contour.reshape(4, 2)

        # Divide each side of the board contour into 9 sections
        top_divisions = np.linspace(top_left, top_right, num=9, endpoint=True)
        right_divisions = np.linspace(top_right, bottom_right, num=9, endpoint=True)
        bottom_divisions = np.linspace(bottom_right, bottom_left, num=9, endpoint=True)
        left_divisions = np.linspace(top_left, bottom_left, num=9, endpoint=True)

        # Connect the dividing points of each side to those of the facing side in the desired order
        left_divisions_flipped = np.flip(left_divisions, axis=0)
        top_divisions_flipped = np.flip(top_divisions, axis=0)
        right_divisions_flipped = np.flip(right_divisions, axis=0)
        bottom_divisions_flipped = np.flip(bottom_divisions, axis=0)

        # Draw the horizontal and vertical lines
        for i in range(len(top_divisions)):
            cv2.line(frame, tuple(map(int, top_divisions_flipped[i])), tuple(map(int, bottom_divisions[i])), (0, 255, 0), 1)
            cv2.line(frame, tuple(map(int, left_divisions[i])), tuple(map(int, right_divisions[i])), (0, 255, 0), 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)

    # Wait for a key press and check if it is the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device and close all windows
cap.release()
cv2.destroyAllWindows()
