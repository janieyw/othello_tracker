import cv2
import numpy as np

# Define green color range
lower_green = np.array([0, 50, 0])
upper_green = np.array([100, 255, 100])

# Capture video from default camera
cap = cv2.VideoCapture(0)

# Loop until user quits
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Apply color thresholding to extract green pixels
    green_mask = cv2.inRange(frame, lower_green, upper_green)
    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the outer rectangle that contains all the contours
    board_rect = cv2.boundingRect(np.concatenate(contours))
    x, y, w, h = board_rect

    # Compute the size of a cell based on the distance between the vertical lines
    num_cols = 9
    col_centers = np.zeros(num_cols, dtype=np.float32)
    col_centers[0] = x + w // num_cols // 2
    for i in range(1, num_cols):
        col_centers[i] = col_centers[i - 1] + w / (num_cols - 1)
    cell_size = int((col_centers[-1] - col_centers[0]) / (num_cols - 1))

    # Compute the size of a cell based on the distance between the horizontal lines
    num_rows = 9
    row_centers = np.zeros(num_rows, dtype=np.float32)
    row_centers[0] = y + h // num_rows // 2
    for i in range(1, num_rows):
        row_centers[i] = row_centers[i - 1] + h / (num_rows - 1)
    cell_size = int((row_centers[-1] - row_centers[0]) / (num_rows - 1))

    # Create a grid of cell centers
    centers_x = np.zeros((num_rows, num_cols), dtype=np.float32)
    centers_y = np.zeros((num_rows, num_cols), dtype=np.float32)
    for i in range(num_rows):
        for j in range(num_cols):
            centers_x[i, j] = col_centers[j]
            centers_y[i, j] = row_centers[i]

    # Apply affine transformation to warp the image into a bird's-eye view
    src_points = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    dst_points = np.float32([[0, 0], [cell_size * (num_cols - 1), 0], [0, cell_size * (num_rows - 1)],
                             [cell_size * (num_cols - 1), cell_size * (num_rows - 1)]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (cell_size * (num_cols - 1), cell_size * (num_rows - 1)))

    # Display the output
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
