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

    # Calculate the maximum area of the contours
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area

    # Iterate through the contours to identify the disks
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Skip contours that are too small to be disks or too large to be the board
        if area < 50 or area == max_area:
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

        # Label the disks based on the percentage of white pixels
        if white_percentage > 60:
            label = "O"
        else:
            label = "X"

        # Draw the label on the original frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(frame, label, (x, y), font, font_scale, color, thickness)

    # Display the original frame with the labeled disks
    cv2.imshow("Tic Tac Toe Game", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources and close the window
cap.release()
cv2.destroyAllWindows()
