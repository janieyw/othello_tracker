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
        print(f"Player 1 leads by {p1_disk_num - p2_disk_num}!")
    elif p1_disk_num < p2_disk_num:  # p2 winning
        print(f"Player 2 leads by {p2_disk_num - p1_disk_num}!")
    else:  # p1_disk_num == p2_disk_num
        print(f"Tie!")

def update_round_result(p1_disk_num, p2_disk_num):
    print_p1_score(p1_disk_num)
    print_p2_score(p2_disk_num)
    print_round_result(p1_disk_num, p2_disk_num)

def print_no_play_message():
    print("TIMEOUT: No play detected!")

def print_no_hand_message():
    print("TIMEOUT: No hand detected!")

def print_game_result(p1_disk_num, p2_disk_num):
    if p1_disk_num > p2_disk_num:  # p1 winning
        print(f"Player 1 wins by {p1_disk_num - p2_disk_num}!")
    elif p1_disk_num < p2_disk_num:  # p2 winning
        print(f"Player 2 wins by {p2_disk_num - p1_disk_num}!")
    else:  # p1_disk_num == p2_disk_num
        print(f"Tie!")

def print_bye_message():
    print_line_separator()
    print("     End of Game!")

def announce_game_end(p1_disk_num, p2_disk_num):
    print_p1_score(p1_disk_num)
    print_p2_score(p2_disk_num)
    print_game_result(p1_disk_num, p2_disk_num)
    print_bye_message()

def announce_no_hand_game_end(grid_colors, p1_disk_num, p2_disk_num):
    print_no_hand_message()
    print_line_separator()
    print_board(grid_colors)
    print_line_separator()
    announce_game_end(p1_disk_num, p2_disk_num)

def announce_no_play_game_end(grid_colors, p1_disk_num, p2_disk_num):
    print_no_play_message()
    print_line_separator()
    print_board(grid_colors)
    print_line_separator()
    announce_game_end(p1_disk_num, p2_disk_num)

def print_wrong_player_warning(right_player_num):
    print(f"Wrong player! Player {right_player_num}'s turn!")