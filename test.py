import math
import player
import time
from game import last_play_detected_time, last_hand_detected_time
from constants import BLACK, WHITE, GREEN, TOTAL_DISK_NUM, GRID_SIZE, TIME_LIMIT
from utils import *
from talker import print_board, print_line_separator, update_round_result, announce_no_hand_game_end, \
    announce_no_play_game_end, announce_game_end

# Initialize the player identification object
player = player.Player()
player_num = None
player_num_stack = []

# Initialize all disk numbers to 0, except for prev_disk_num
p1_disk_num, p2_disk_num = 0, 0
disk_num = 0
prev_disk_num = -1

# Initialize grid_colors with all '-'
grid_colors = [[GREEN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

cap = cv2.VideoCapture(0)  # Use iPhone as webcam

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Count the number of disks on the board
    total_disk_num = p1_disk_num + p2_disk_num

    # Identify the current player
    player.get_current_player(frame)

    # Draw the current player number on the frame
    player_num = player.get_current_player_num()

    p1_disk_num, p2_disk_num = reset_player_disk_num()

    # # Check if the next player number matches the topmost element of the stack
    # if player_num is not None and player_num == player_num_stack[-1]:
    #     # Remove the topmost element from the stack
    #     player_num_stack.pop()
    # else:
    #     # Print the "Wrong player turn!" message
    #     print("Wrong player turn!")
    #
    # # Add the current player number to the stack
    # player_num_stack.append(player_num)

    # Identify the current player
    player.get_current_player(frame)

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply morphological operations to fill any gaps in the masks
    kernel = np.ones((5, 5), np.uint8)
    green_mask = get_green_mask(hsv, kernel)
    black_mask = get_black_mask(hsv, kernel)
    white_mask = get_white_mask(hsv, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Find the largest contour
    largest_contour = find_largest_contour(contours)

    intersection_points = []

    # Initialize a two-dimensional array to store the corner points of each grid cell
    grid_cells = [['-' for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]

    # Draw a green outline around the largest contour
    if largest_contour is not None:
        # cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)

        top_left, top_right, bottom_right, bottom_left = largest_contour.reshape(4, 2)

        # intersection_points.append((top_left[0], top_left[1]))
        # intersection_points.append((top_right[0], top_right[1]))
        # intersection_points.append((bottom_right[0], bottom_right[1]))
        # intersection_points.append((bottom_left[0], bottom_left[1]))

        ver_lines = set()
        hor_lines = set()

        # Calculate the angles of the sides
        top_angle = math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
        bottom_angle = math.atan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0])
        left_angle = math.atan2(top_left[1] - bottom_left[1], top_left[0] - bottom_left[0])
        right_angle = math.atan2(top_right[1] - bottom_right[1], top_right[0] - bottom_right[0])

        # Calculate the length and angle of each side of the quadrilateral
        top_side_length = np.linalg.norm(np.array(top_right) - np.array(top_left))
        top_side_angle = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0])

        right_side_length = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        right_side_angle = np.arctan2(bottom_right[1] - top_right[1], bottom_right[0] - top_right[0])

        bottom_side_length = np.linalg.norm(np.array(bottom_left) - np.array(bottom_right))
        bottom_side_angle = np.arctan2(bottom_left[1] - bottom_right[1], bottom_left[0] - bottom_right[0])

        left_side_length = np.linalg.norm(np.array(top_left) - np.array(bottom_left))
        left_side_angle = np.arctan2(top_left[1] - bottom_left[1], top_left[0] - bottom_left[0])

        top_divisions = np.linspace(top_left, top_right, num=GRID_SIZE + 1, endpoint=True)
        right_divisions = np.linspace(top_right, bottom_right, num=GRID_SIZE + 1, endpoint=True)
        bottom_divisions = np.linspace(bottom_right, bottom_left, num=GRID_SIZE + 1, endpoint=True)
        left_divisions = np.linspace(bottom_left, top_left, num=GRID_SIZE + 1, endpoint=True)

        left_divisions_flipped = np.flip(left_divisions, axis=0)
        top_divisions_flipped = np.flip(top_divisions, axis=0)
        right_divisions_flipped = np.flip(right_divisions, axis=0)
        bottom_divisions_flipped = np.flip(bottom_divisions, axis=0)

        # Draw lines connecting corresponding segments on opposite sides
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
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

                # # Draw lines connecting opposite sides
                # cv2.line(frame, (int(left_top[0]), int(left_top[1])), (int(right_top[0]), int(right_top[1])), color=(0, 255, 0),
                #          thickness=2)
                # cv2.line(frame, (int(left_bottom[0]), int(left_bottom[1])), (int(right_bottom[0]), int(right_bottom[1])), color=(0, 255, 0),
                #          thickness=2) # Left side
                # cv2.line(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_left[0]), int(bottom_left[1])), color=(0, 255, 0),
                #          thickness=2)
                # cv2.line(frame, (int(top_right[0]), int(top_right[1])), (int(bottom_right[0]), int(bottom_right[1])), color=(0, 255, 0),
                #          thickness=2)

                # Draw lines connecting opposite sides
                ver_line1 = (int(left_top[0]), int(left_top[1]), int(right_top[0]), int(right_top[1]))
                ver_line2 = (int(left_bottom[0]), int(left_bottom[1]), int(right_bottom[0]), int(right_bottom[1]))
                hor_line1 = (int(top_left[0]), int(top_left[1]), int(bottom_left[0]), int(bottom_left[1]))
                hor_line2 = (int(top_right[0]), int(top_right[1]), int(bottom_right[0]), int(bottom_right[1]))

                ver_lines.update([ver_line1, ver_line2])
                hor_lines.update([hor_line1, hor_line2])

                # Add outer points of the quadrilateral if they haven't been added yet
                outer_points = [(int(top_left[0]), int(top_left[1])),
                                (int(top_right[0]), int(top_right[1])),
                                (int(bottom_left[0]), int(bottom_left[1])),
                                (int(bottom_right[0]), int(bottom_right[1])),
                                (int(left_top[0]), int(left_top[1])),
                                (int(left_bottom[0]), int(left_bottom[1])),
                                (int(right_top[0]), int(right_top[1])),
                                (int(right_bottom[0]), int(right_bottom[1]))]

                for outer_point in outer_points:
                    if all(np.linalg.norm(np.array(outer_point) - np.array(point)) > 5 for point in
                           intersection_points):
                        intersection_points.append(outer_point)

        for v_line in ver_lines:
            for h_line in hor_lines:
                start_v = (v_line[0], v_line[1])
                end_v = (v_line[2], v_line[3])
                start_h = (h_line[0], h_line[1])
                end_h = (h_line[2], h_line[3])
                # cv2.line(frame, start_v, end_v, color=(0, 255, 0), thickness=2)
                # cv2.line(frame, start_h, end_h, color=(0, 255, 0), thickness=2)
                intersection_point = compute_intersection(v_line, h_line)
                if intersection_point is not None and all(
                        np.linalg.norm(np.array(intersection_point) - np.array(point)) > 5 for point in
                        intersection_points):
                    intersection_points.append(intersection_point)

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
                    # Create binary masks for the current cell rectangle
                    cell_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    # Draw the contours of the current cell onto the cell_mask and cell_rect
                    cv2.drawContours(cell_mask, [np.array([top_left, top_right, bottom_right, bottom_left])], 0,
                                     (255, 255, 255), -1)
                    cv2.drawContours(cell_rect, [np.array([top_left, top_right, bottom_right, bottom_left])], 0,
                                     (255, 255, 255), -1)
                    # Apply the masks to the color masks
                    green_mask_cell = cv2.bitwise_and(green_mask, cell_mask)
                    black_mask_cell = cv2.bitwise_and(black_mask, cell_mask)
                    white_mask_cell = cv2.bitwise_and(white_mask, cell_mask)

                    # Determine the dominant color
                    color = determine_dominant_color(frame, green_mask_cell, black_mask_cell, white_mask_cell)

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