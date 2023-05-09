import player
import time
from game import last_play_detected_time, last_hand_detected_time
from constants import BLACK, WHITE, GREEN, TOTAL_DISK_NUM, GRID_SIZE, TIME_LIMIT
from board import *
from talker import *

# Initialize the player identification object
player = player.Player()

# Initialize all disk numbers to 0, except for prev_disk_num
p1_disk_num, p2_disk_num, total_disk_num = 0, 0, 0
prev_disk_num = -1
prev_player_num = None
prev_grid_colors_need_update = False

# Initialize grid_colors with all '-'
grid_colors = [[GREEN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
prev_grid_colors = None
player_num_stack = [2]

cap = cv2.VideoCapture(0)  # Use iPhone as webcam

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    p1_disk_num, p2_disk_num = reset_player_disk_num()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply morphological operations to fill any gaps in the masks
    kernel = np.ones((5, 5), np.uint8)
    green_mask, black_mask, white_mask = get_color_masks(hsv, kernel)

    # Find the largest contour
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)

    intersection_points = []

    # Initialize a two-dimensional array to store the corner points of each grid cell
    grid_cells = [['-' for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]

    # Identify the current player
    player_num = player.get_current_player_num(frame)

    if player_num is not None:
        # Update prev_player_num to current player_num
        prev_player_num = player_num_stack[0]

        # Check if player_num is the same as prev_player_num, and print "wrong player!" if so
        if player_num == prev_player_num:
            right_player_num = player.get_right_player_num(prev_player_num)
            print_wrong_player_warning(right_player_num)
        else:
            player_num_stack.pop()
            player_num_stack.append(player_num)
            # update grid colors

    # Draw a green outline around the largest contour
    if largest_contour is not None:

        intersection_points = process_contour(largest_contour, GRID_SIZE)
        # Display intersection points in gradient, while incrementing blue value and decrementing red value
        display_in_gradient(frame, intersection_points, 0, 255)

        if len(intersection_points) == 81:
            # Loop through each row and column to add the four corner points for each cell
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    top_left, top_right, bottom_left, bottom_right = define_corner_points(intersection_points, i, j)
                    grid_cells[i][j] = [top_left, top_right, bottom_left, bottom_right]

                    # Draw the quadrilateral
                    draw_quadrilateral(frame, top_left, bottom_right)

                    # Define the masks for the current cell rectangle
                    cell_rect = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cell_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    # Draw the contours of the current cell onto the cell_mask and cell_rect
                    cv2.drawContours(cell_mask, [np.array([top_left, top_right, bottom_right, bottom_left])], 0,
                                     (255, 255, 255), -1)
                    cv2.drawContours(cell_rect, [np.array([top_left, top_right, bottom_right, bottom_left])], 0,
                                     (255, 255, 255), -1)

                    color = determine_dominant_color(frame, cell_mask, green_mask, black_mask, white_mask)
                    grid_colors[i][j] = color

                    draw_disk(frame, color, top_left, bottom_right)

        if prev_grid_colors_need_update:
            prev_grid_colors = grid_colors

        total_disk_num, p1_disk_num, p2_disk_num = count_disks(grid_colors)

        if prev_grid_colors is not None:
            prev_total_disk_num, prev_p1_disk_num, prev_p2_disk_num = count_disks(prev_grid_colors)

            if total_disk_num != prev_total_disk_num + 1:
                print_one_disk_only_warning()

            if not disk_added_to_empty_cell(prev_grid_colors, grid_colors):
                print_add_to_empty_cell_warning()

        # End the game if no hand has been detected or no disk has been added for 30 seconds
        if time.time() - last_hand_detected_time > TIME_LIMIT:
            player_num = None
            announce_no_hand_game_end(grid_colors, p1_disk_num, p2_disk_num)
            break

        # # End the game if no disk has been added for 30 seconds
        # if total_disk_num == prev_disk_num and time.time() - last_play_detected_time > TIME_LIMIT:
        #     player_num = None
        #     announce_no_play_game_end(grid_colors, p1_disk_num, p2_disk_num)
        #     break

        # Display the current player number on the frame if player_num is not None
        if player_num is not None:
            last_hand_detected_time = time.time()
            display_player_num(frame, player_num)

        # Check for 'space' key press to print out grid_colors
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print_grid_colors_for_space(grid_colors, p1_disk_num, p2_disk_num)

        # Check for 's' key press to analyze
        if cv2.waitKey(1) & 0xFF == ord('s'):
            prev_grid_colors_need_update = True

        # Update the prev_disk_num variable
        prev_disk_num = total_disk_num

    # Display the resulting frame
    cv2.imshow('Othello Tracker', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        announce_game_end(p1_disk_num, p2_disk_num)
        break

cap.release()
cv2.destroyAllWindows()