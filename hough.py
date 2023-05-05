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
    cv2.imshow('Othello Tracker', frame)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# def detect_disks(frame):
#     # Convert the color frame to HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Define the lower and upper bounds of the green color in HSV color space
#     lower_green = np.array([40, 50, 50])
#     upper_green = np.array([80, 255, 255])
#
#     # Threshold the HSV image to get a binary mask of the green area
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#
#     # Apply morphology operations to remove noise and fill gaps in the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # Apply the mask to the original frame to get the segmented image
#     segmented = cv2.bitwise_and(frame, frame, mask=mask)
#
#     # Convert the segmented image to grayscale
#     gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
#
#     # Apply thresholding to the grayscale image
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#
#     # Find contours in the thresholded image
#     contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Iterate through the contours to identify the disks
#     for contour in contours:
#         # Calculate the area of the contour
#         area = cv2.contourArea(contour)
#
#         # Skip contours that are too small to be disks
#         if area < 50:
#             continue
#
#         # Calculate the circularity of the contour
#         perimeter = cv2.arcLength(contour, True)
#         circularity = 4 * np.pi * area / perimeter ** 2
#
#         # Skip contours that are not circular enough to be disks
#         if circularity < 0.5:
#             continue
#
#         # Find the bounding box of the contour
#         x, y, w, h = cv2.boundingRect(contour)
#
#         # Extract the ROI corresponding to the contour
#         roi = frame[y:y + h, x:x + w]
#
#         # Convert the ROI to grayscale
#         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#
#         # Apply thresholding to the grayscale ROI
#         thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#
#         # Calculate the percentage of white pixels in the ROI
#         white_pixels = np.count_nonzero(thresh_roi == 255)
#         total_pixels = thresh_roi.size
#         white_percentage = white_pixels / total_pixels * 100
#
#         # Label the disk as white or black based on the percentage of white pixels
#         if white_percentage > 50:
#             color = (0, 0, 255)  # red for white disks
#         else:
#             color = (0, 255, 0)  # green for black disks
#
#         # Draw a circle around the contour
#         cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), int(w / 2), color, 2)
#
# def detect_grid(frame):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Apply adaptive thresholding to the grayscale image
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#
#     # Find contours in the thresholded image
#     contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Create a list to store the corners of the grid
#     corners = []
#
#     # Iterate through the contours to identify the corners of the grid
#     for contour in contours:
#         # Calculate the area of the contour
#         area = cv2.contourArea(contour)
#
#         # Skip contours that are too small to be corners
#         if area < 100:
#             continue
#
#         # Find the centroid of the contour
#         M = cv2.moments(contour)
#         cx = int(M['m10'] / M['m00'])
#         cy = int(M['m01'] / M['m00'])
#
#         # Append the centroid to the list of corners
#         corners.append((cx, cy))
#
#         # Draw a circle around the centroid
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
#
#     # Sort the corners in order of their x and y coordinates
#     corners = sorted(corners, key=lambda x: (x[0], x[1]))
#
#     # Calculate the distance between the first and last corners in the x and y directions
#     x_dist = corners[-1][0] - corners[0][0]
#     y_dist = corners[-1][1] - corners[0][1]
#
#     # Calculate the size of each cell in the grid
#     cell_size = min(x_dist, y_dist) / 8
#
#     # Create a list to store the cells of the grid
#     cells = []
#
#     # Iterate through the rows and columns of the grid to extract each cell
#     for i in range(8):
#         for j in range(8):
#             # Calculate the coordinates of the top-left corner of the cell
#             x = int(corners[0][0] + j * cell_size)
#             y = int(corners[0][1] + i * cell_size)
#
#             # Extract the ROI corresponding to the cell
#             roi = frame[y:y + int(cell_size), x:x + int(cell_size)]
#
#             # Append the cell to the list of cells
#             cells.append(roi)
#
#     return cells
#
# def main():
#     # Initialize the camera and set the resolution
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#     # Loop through the frames in the video
#     while True:
#         # Read a frame from the video
#         ret, frame = cap.read()
#
#         # If the frame was not successfully read, break out of the loop
#         if not ret:
#             break
#
#         # Process the frame to detect the disks and grid
#         detect_disks(frame)
#         detect_grid(frame)
#
#         # Display the processed frame
#         cv2.imshow('Othello Tracker', frame)
#
#         # Exit if the user presses 'q'
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#     # Release the video capture and close all windows
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     main()