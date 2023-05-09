class Talker:
    @staticmethod
    def print_board(grid_colors):
        for row in grid_colors:
            print('  '.join(str(elem) for elem in row))  # print(' '.join(row))

    @staticmethod
    def print_line_separator():
        print("----------------------")

    @staticmethod
    def print_p1_score(p1_disk_num):
        print(f"Player 1 score: {p1_disk_num:2d}")

    @staticmethod
    def print_p2_score(p2_disk_num):
        print(f"Player 2 score: {p2_disk_num:2d}")

    @staticmethod
    def print_round_result(p1_disk_num, p2_disk_num):
        if p1_disk_num > p2_disk_num:  # p1 winning
            print(f"Player 1 leads by {p1_disk_num - p2_disk_num}!")
        elif p1_disk_num < p2_disk_num:  # p2 winning
            print(f"Player 2 leads by {p2_disk_num - p1_disk_num}!")
        else:  # p1_disk_num == p2_disk_num
            print(f"Tie!")

    @staticmethod
    def update_round_result(p1_disk_num, p2_disk_num):
        Talker.print_p1_score(p1_disk_num)
        Talker.print_p2_score(p2_disk_num)
        Talker.print_round_result(p1_disk_num, p2_disk_num)

    @staticmethod
    def print_no_play_message():
        print("TIMEOUT: No play detected!")

    @staticmethod
    def print_no_hand_message():
        print("TIMEOUT: No hand detected!")

    @staticmethod
    def print_game_result(p1_disk_num, p2_disk_num):
        if p1_disk_num > p2_disk_num:  # p1 winning
            print(f"Player 1 wins by {p1_disk_num - p2_disk_num}!")
        elif p1_disk_num < p2_disk_num:  # p2 winning
            print(f"Player 2 wins by {p2_disk_num - p1_disk_num}!")
        else:  # p1_disk_num == p2_disk_num
            print(f"Tie!")

    @staticmethod
    def print_bye_message():
        Talker.print_line_separator()
        print("     End of Game!")

    @staticmethod
    def announce_game_end(p1_disk_num, p2_disk_num):
        Talker.print_p1_score(p1_disk_num)
        Talker.print_p2_score(p2_disk_num)
        Talker.print_game_result(p1_disk_num, p2_disk_num)
        Talker.print_bye_message()

    @staticmethod
    def announce_no_hand_game_end(grid_colors, p1_disk_num, p2_disk_num):
        Talker.print_no_hand_message()
        Talker.print_line_separator()
        Talker.print_board(grid_colors)
        Talker.print_line_separator()
        Talker.announce_game_end(p1_disk_num, p2_disk_num)

    @staticmethod
    def announce_no_play_game_end(grid_colors, p1_disk_num, p2_disk_num):
        Talker.print_no_play_message()
        Talker.print_line_separator()
        Talker.print_board(grid_colors)
        Talker.print_line_separator()
        Talker.announce_game_end(p1_disk_num, p2_disk_num)

    @staticmethod
    def print_wrong_player_warning(right_player_num):
        if right_player_num is not None:
            print(f"WARNING: Player {right_player_num}'s turn!")

    @staticmethod
    def print_one_disk_only_warning():
        print("WARNING: Only 1 disk at a time!")

    @staticmethod
    def print_add_to_empty_cell_warning():
        print("WARNING: Place disk only in an empty cell!")

    @staticmethod
    def print_grid_colors_for_space(grid_colors, p1_disk_num, p2_disk_num):
        Talker.print_board(grid_colors)
        Talker.print_line_separator()
        Talker.update_round_result(p1_disk_num, p2_disk_num)
        Talker.print_line_separator()
