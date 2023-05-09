import board, player
import cv2, numpy as np
import time
from game import last_play_detected_time, last_hand_detected_time
from constants import *
from talker import Talker

player = player.Player()
detector = board.BoardDetector()

# Initialize all disk numbers to 0, except for prev_disk_num
p1_disk_num, p2_disk_num, total_disk_num = 0, 0, 0
# prev_disk_num = -1

prev_player_num = 2
prev_grid_colors = None
prev_grid_colors_need_update = False

cap = cv2.VideoCapture(0)  # Use iPhone as webcam

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Reset player disk numbers
    p1_disk_num, p2_disk_num = detector.reset_player_disk_num()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply morphological operations to fill any gaps in the masks
    kernel = np.ones((5, 5), np.uint8)
    green_mask, black_mask, white_mask = detector.get_color_masks(hsv, kernel)

    # Find the largest contour
    largest_contour = detector.find_largest_contour(green_mask)

    # Create an empty list to store intersection points
    intersection_points = []

    # Initialize two 2D array to detect grid cells and grid colors, respectively
    grid_cells = grid_colors = [[GREEN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    # Identify the current player
    player_num = player.get_current_player_num(frame)

    if player_num != -1:
        if prev_player_num == player_num:
            Talker.display_wrong_player_warning(frame, right_player_num=player.get_right_player_num(prev_player_num))
        prev_player_num = player_num

    # Check if the largest contour has been detected
    if largest_contour is not None:

        # Extract intersection points from the largest contour
        intersection_points = detector.extract_intersection_points(largest_contour, GRID_SIZE)

        # Display intersection points in gradient, while incrementing blue value and decrementing red value
        detector.display_in_gradient(frame, intersection_points, 0, 255)

        # Check if all intersection points have been detected
        if len(intersection_points) == 81:
            # Iterate over each cell in the grid
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):

                    # Define corner points of the current cell
                    top_left, top_right, bottom_left, bottom_right = detector.define_corner_points(intersection_points, i, j)

                    # Store the corner points of the current cell
                    grid_cells[i][j] = [top_left, top_right, bottom_left, bottom_right]

                    # Draw the current cell on the frame
                    detector.draw_grid_cell(frame, top_left, bottom_right)

                    # Create cell mask for the grid cell based on its four corners
                    cell_mask = detector.create_cell_mask(frame, top_left, top_right, bottom_right, bottom_left)

                    # Determine the dominant color in the grid cell for disk detection
                    color = detector.determine_dominant_color(frame, cell_mask, green_mask, black_mask, white_mask)

                    # Assign the dominant color to the corresponding cell in grid_colors
                    grid_colors[i][j] = color

                    # Draw the disk on the grid cell in the frame
                    detector.draw_disk(frame, color, top_left, bottom_right)

        # Update the previous grid_colors if needed
        if prev_grid_colors_need_update:
            prev_grid_colors = grid_colors

        # Count the total number of disks on the board, as well as the number of disks for each player
        total_disk_num, p1_disk_num, p2_disk_num = detector.count_disks(grid_colors)

        # Check if the previous grid_colors exist & Update the previous disk counts accordingly
        if prev_grid_colors is not None:
            prev_total_disk_num, prev_p1_disk_num, prev_p2_disk_num = detector.count_disks(prev_grid_colors)

            # Display a warning if more than one disk was added on the current turn
            if total_disk_num > prev_total_disk_num + 1:
                Talker.display_one_disk_only_warning(frame)

            # Display a warning if no disks were added on the current turn
            if total_disk_num <= prev_total_disk_num:
                Talker.display_add_to_empty_cell_warning(frame)

            # Display a warning if the wrong color was added on the current turn
            else:
                disk_added_cell = detector.get_disk_added_cell(prev_grid_colors, grid_colors)
                if detector.wrong_color_added(prev_player_num, disk_added_cell):
                    Talker.display_wrong_color_warning(frame)

        # End the game if no hand has been detected or no disk has been added for 30 seconds
        if time.time() - last_hand_detected_time > TIME_LIMIT:
            player_num = None
            Talker.announce_no_hand_game_end(grid_colors, p1_disk_num, p2_disk_num)
            break

        # # End the game if no disk has been added for 30 seconds
        # if total_disk_num == prev_disk_num and time.time() - last_play_detected_time > TIME_LIMIT:
        #     player_num = None
        #     announce_no_play_game_end(grid_colors, p1_disk_num, p2_disk_num)
        #     break

        # Display the current player number on the frame if player_num is not None
        if player_num is not None:
            last_hand_detected_time = time.time()
            Talker.display_player_num(frame, player_num)

        # Check for 'space' key press to print out grid_colors
        if cv2.waitKey(1) & 0xFF == ord(' '):
            Talker.print_grid_colors_for_space(grid_colors, p1_disk_num, p2_disk_num)

        # Check for 's' key press to analyze
        if cv2.waitKey(1) & 0xFF == ord('s'):
            prev_grid_colors_need_update = True

        # # Update the prev_disk_num variable
        # prev_disk_num = total_disk_num

    # Display the resulting frame
    cv2.imshow('Othello Tracker', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q') or total_disk_num >= TOTAL_DISK_NUM:
        Talker.announce_game_end(p1_disk_num, p2_disk_num)
        break

cap.release()
cv2.destroyAllWindows()