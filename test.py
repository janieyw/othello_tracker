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

# Initialize grid_colors with all '-'
grid_colors = [[GREEN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

cap = cv2.VideoCapture(0)  # Use iPhone as webcam

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Count the number of disks on the board
    total_disk_num = p1_disk_num + p2_disk_num
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

    # Check if player_num is the same as prev_player_num, and print "wrong player!" if so
    if player_num == prev_player_num:
        right_player_num = player.get_right_player_num(prev_player_num)
        print_wrong_player_warning(right_player_num)

    # Update prev_player_num to current player_num
    prev_player_num = player_num

    # Draw a green outline around the largest contour
    if largest_contour is not None:
        # cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)
        top_left, top_right, bottom_right, bottom_left = largest_contour.reshape(4, 2)

        # Calculate the angles and lengths of the sides
        top_side_length, top_side_angle = calculate_side_properties(top_left, top_right)
        right_side_length, right_side_angle = calculate_side_properties(top_right, bottom_right)
        bottom_side_length, bottom_side_angle = calculate_side_properties(bottom_right, bottom_left)
        left_side_length, left_side_angle = calculate_side_properties(bottom_left, top_left)

        # Divide each side into GRID_SIZE equal parts
        top_divisions = divide_side_into_segments(top_left, top_right, GRID_SIZE)
        right_divisions = divide_side_into_segments(top_right, bottom_right, GRID_SIZE)
        bottom_divisions = divide_side_into_segments(bottom_right, bottom_left, GRID_SIZE)
        left_divisions = divide_side_into_segments(bottom_left, top_left, GRID_SIZE)

        # Flip the left, top, right, and bottom divisions
        left_divisions_flipped = flip_divisions(left_divisions)
        top_divisions_flipped = flip_divisions(top_divisions)
        right_divisions_flipped = flip_divisions(right_divisions)
        bottom_divisions_flipped = flip_divisions(bottom_divisions)

        intersection_points = find_intersection_points(top_divisions_flipped, bottom_divisions, left_divisions_flipped, right_divisions)
        intersection_points = sorted(intersection_points, key=lambda p: (p[1], p[0]))

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

        # Loop through the grid_colors array
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid_colors[i][j] == BLACK:
                    p1_disk_num += 1
                elif grid_colors[i][j] == WHITE:
                    p2_disk_num += 1

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

        # Print out the color information for each grid cell
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print_board(grid_colors)
            print_line_separator()
            update_round_result(p1_disk_num, p2_disk_num)
            print_line_separator()

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