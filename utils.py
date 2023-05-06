import cv2

def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if den == 0:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
    if ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (int(x), int(y))
    else:
        return None

def find_largest_contour(contours):
    largest_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            largest_contour = approx
            break
    return largest_contour

def display_in_gradient(frame, intersection_points, blue_value, red_value):
    for p in range(len(intersection_points)):
        if p == 0:
            radius = 20
        else:
            radius = 8

        # Draw the point with the new color
        cv2.circle(frame, intersection_points[p], radius=radius, color=(blue_value, 0, red_value), thickness=-1)

        # Increment and decrement color value by a fixed amount
        blue_value += 3
        red_value -= 3

def display_player_num(frame, player_num):
    cv2.putText(frame, f"Player {player_num}", (25, 65), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

def print_board(grid_colors):
    for row in grid_colors:
        print('  '.join(str(elem) for elem in row))  # print(' '.join(row))

def print_line_separator():
    print("----------------------")

def print_p1_score(p1_disk_num):
    print(f"Player 1 score: {p1_disk_num:2d}")

def print_p2_score(p2_disk_num):
    print(f"Player 2 score: {p2_disk_num:2d}")

def print_round_result(p1_disk_num, p2_disk_num):
    if p1_disk_num > p2_disk_num:  # p1 winning
        print(f"Player 1 is winning by {p1_disk_num - p2_disk_num}!")
    elif p1_disk_num < p2_disk_num:  # p2 winning
        print(f"Player 2 is winning by {p2_disk_num - p1_disk_num}!")
    else:  # p1_disk_num == p2_disk_num
        print(f"Tie!")

def print_no_play_message():
    print("TIMEOUT: No play detected!")

def print_no_hand_message():
    print("TIMEOUT: No hand detected!")

def print_game_result(p1_disk_num, p2_disk_num):
    if p1_disk_num > p2_disk_num:  # p1 winning
        print(f"Player 1 wins by {p1_disk_num - p2_disk_num}!")
    elif p1_disk_num < p2_disk_num:  # p2 winning
        print(f"Player 2 is winning by {p2_disk_num - p1_disk_num}!")
    else:  # p1_disk_num == p2_disk_num
        print(f"Tie!")

def print_bye_message():
    print_line_separator()
    print("     End of Game!")

def end_game(p1_disk_num, p2_disk_num):
    print_game_result(p1_disk_num, p2_disk_num)
    print_bye_message()