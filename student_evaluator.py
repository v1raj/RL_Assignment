import importlib
import numpy as np
import os
from TicTacToe import *
import pandas as pd
import stopit


rand_seed_list_0 = [6969, 4123, 3781, 21461, 26279, 26741, 10320, 30119, 14617, 777]  # list of random seens being used for checking
rand_seed_list_1 = [12224, 20757, 1428, 8198, 5183, 8701, 21631, 17385, 27501, 1408]


def play_full_game(playerSQN, smartness):
    """Plays a full game of TicTacToe"""

    game = TicTacToe(smartness, playerSQN)
    game.play_game()
    game.print_board()
    reward = game.get_reward()
    return reward


def check(submission):
    """Checks the submission"""

    # Dynamically import the module
    module_name = f"{submission}"
    my_module = importlib.import_module(module_name)

    marks = {'name': module_name}
    total_marks = 0

    print("\n_____________Smartness 0_____________\n")
    for i, seed in enumerate(rand_seed_list_0):
        random.seed(seed)
        playerSQN = my_module.PlayerSQN()
        smartness = 0
        decision = ''
        with stopit.ThreadingTimeout(10) as context_manager:    # ensuring time limit of 10 seconds per case
            game_reward = play_full_game(playerSQN, smartness)
            marks[i] = game_reward
            total_marks += 3 if game_reward == 1 else (1 if game_reward == 0 else 0.5)
            decision = 'WIN' if game_reward == 1 else ('TIE' if game_reward == 0 else 'LOSS')
        if context_manager.state == context_manager.TIMED_OUT:
            decision = 'TIME OUT'
        del playerSQN
        print(f"Smartness = 0, seed = {seed}, Result = {decision}")
    
    print("\n_____________Smartness 0.8_____________\n")
    for i, seed in enumerate(rand_seed_list_1):
        random.seed(seed)
        playerSQN = my_module.PlayerSQN()
        smartness = 0.8
        decision = ''
        with stopit.ThreadingTimeout(10) as context_manager:    # ensuring time limit of 10 seconds per case
            game_reward = play_full_game(playerSQN, smartness)
            marks[10 + i] = game_reward
            total_marks += 3 if game_reward == 1 else (1 if game_reward == 0 else 0.5)
            decision = 'WIN' if game_reward == 1 else ('TIE' if game_reward == 0 else 'LOSS')
        if context_manager.state == context_manager.TIMED_OUT:
            decision = 'TIME OUT'
        del playerSQN
        print(f"Smartness = 0.8, seed = {seed}, Result = {decision}")

    marks['total'] = total_marks
    return {'name': marks['name'], 'total': marks['total']}


if __name__ == "__main__":
    # get list of submissions
    your_id = '2020A7PS1011G'
    d = check(your_id)
    print(f"Final results: {d['name']} achieved score = {d['total']} out of 60")