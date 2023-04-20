import cv2
import numpy as np

# Initialize the camera and set the resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the color frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the green color in HSV color space
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get a binary mask of the green area
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphology operations to remove noise and fill gaps in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original frame to get the segmented image
    segmented = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the segmented image to grayscale
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to the grayscale image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough transform to the edge map
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Filter the lines based on their length, angle, and proximity to each other
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if length > 100 and abs(angle) > 80 and abs(angle) < 100:
                filtered_lines.append(line)
            elif length > 100 and abs(angle) < 10:
                filtered_lines.append(line)

    # Find the intersection points of the remaining lines
    corners = []
    for i in range(len(filtered_lines)):
        for j in range(i + 1, len(filtered_lines)):
            x1, y1, x2, y2 = filtered_lines[i][0]
            x3, y3, x4, y4 = filtered_lines[j][0]
            d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if d != 0:
                pt = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d, \
                     ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
                corners.append(pt)

    # Draw the lines and corners on the original image
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for pt in corners:
        x, y = pt
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the resulting image
    cv2.imshow('Game Board', frame)

    # Exit the program if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
