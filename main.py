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

    # Apply thresholding to the grayscale image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours to identify the disks
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Skip contours that are too small to be disks
        if area < 50:
            continue

        # Calculate the circularity of the contour
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / perimeter ** 2

        # Skip contours that are not circular enough to be disks
        if circularity < 0.5:
            continue

        # Find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the ROI corresponding to the contour
        roi = frame[y:y + h, x:x + w]

        # Convert the ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to the grayscale ROI
        thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Calculate the percentage of white pixels in the ROI
        white_pixels = np.count_nonzero(thresh_roi == 255)
        total_pixels = thresh_roi.size
        white_percentage = white_pixels / total_pixels * 100

        # Label the disk as white or black based on the percentage of white pixels
        if white_percentage > 50:
            color = (0, 0, 255)  # red for white disks
        else:
            color = (0, 255, 0)  # green for black disks

        # Draw a circle around the contour
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), int(w / 2), color, 2)

    # Display the frame with the detected disks
    cv2.imshow('frame', frame)

    # Apply Hough transform to the edge map to detect the grid
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Draw the detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the original frame with the detected disks and grid
    cv2.imshow('frame', frame)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
