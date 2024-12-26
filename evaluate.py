# print("\033[31mThis is red text\033[0m")
# print("\033[32mThis is green text\033[0m")

import importlib
import numpy as np
import os
from TicTacToe import *


rand_seed_list = [42, 1233, 41]  # list of random seens being used for checking


def play_full_game(playerSQN, smartness):
    """Plays a full game of TicTacToe"""

    game = TicTacToe(smartness, playerSQN)
    game.play_game()
    reward = game.get_reward()

    if reward == 1:
        print("\033[32mTest passed\033[0m")
    elif reward == 0:
        print("\033[32mTIE\033[0m")
    else:
        print("\033[31mTest failed\033[0m")

    return reward


def print_marks(smartness, total_reward):
    """Prints the marks based on the total reward"""
    marks = 0
    print(f"Total reward: {total_reward} for smartness: {smartness}")

    if total_reward > 0:
        print("\033[32mMarks 4\033[0m")
        marks = 4
    elif total_reward == 0:
        print("\033[32mMarks 2.5\033[0m")
        marks = 2.5
    elif total_reward < 0:
        print("\033[32mMarks 1.5\033[0m")
        marks = 1.5
    else:
        print("\033[31mTest failed\033[0m")

    return marks


def check(submission):
    """Checks the submission"""

    # Dynamically import the module
    module_name = f"{submission}"
    my_module = importlib.import_module(module_name)

    total_marks = 0

    smartness_0_reward = 0  # Total reward when playing against smartness = 0

    print("\n_____________Smartness 0_____________\n")
    for seed in rand_seed_list:
        random.seed(seed)
        playerSQN = my_module.PlayerSQN()
        print(f"Testing full game at smartness = 0, seed = {seed}")
        smartness = 0
        game_reward = play_full_game(playerSQN, smartness)
        smartness_0_reward += game_reward
        del playerSQN

    total_marks += print_marks(0, smartness_0_reward)


    smartness_0_8_reward = 0  # Total reward when playing against smartness = 0.8
    print("\n_____________Smartness 0.8_____________\n")
    for seed in rand_seed_list:
        random.seed(seed)
        playerSQN = my_module.PlayerSQN()
        print(f"Testing full game at smartness = 0.8, seed = {seed}")
        smartness = 0.8
        game_reward = play_full_game(playerSQN, smartness)
        smartness_0_8_reward += game_reward
        del playerSQN

    total_marks += print_marks(0.8, smartness_0_8_reward)


    print(f"\n\nTotal marks: {total_marks} out of 8")


if __name__ == "__main__":
    # get list of submissions
    current_directory = os.getcwd()
    files = [
        f
        for f in os.listdir(current_directory)
        if os.path.isfile(os.path.join(current_directory, f)) and f.startswith("20") and f.endswith(".py")
    ]

    for submission in files:
        check(submission[:-3])
