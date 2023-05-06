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
        print(f"Player 2 leads by {p2_disk_num - p1_disk_num}!")
    else:  # p1_disk_num == p2_disk_num
        print(f"Tie!")

def print_bye_message():
    print_line_separator()
    print("     End of Game!")

def end_game(p1_disk_num, p2_disk_num):
    print_game_result(p1_disk_num, p2_disk_num)
    print_bye_message()