import random

class TicTacToe:
    def __init__(self, smartMovePlayer1=0, playerSQN=None):
        """
        Initializes a TicTacToe game.

        Parameters:
        smartMovePlayer1 (float): The probability that Player 1 will make a smart move. Should be between 0 and 1.
                                  During a smart move, Player 1 attempts to win or block the opponent.
                                  During a non-smart move, Player 1 uniformly randomly selects a valid action.
        playerSQN (PlayerSQN): The player that controls Player 2, typically an instance of the PlayerSQN class.

        Attributes:
        board (list): A list of 9 elements representing the current game board.
        current_winner (int or None): Tracks the winner of the game. None if no player has won yet.
        smartMovePlayer1 (float): Probability of Player 1 making a smart move.
        playerSQN (PlayerSQN): Player 2, which will eventually be implemented as a Shallow Q-Network.
        """
        self.board = [0] * 9  # Board is represented as a list of 9 elements
        self.current_winner = None
        assert 0 <= smartMovePlayer1 <= 1, "Probability of Smart Move must lie between 0 and 1"
        self.smartMovePlayer1 = smartMovePlayer1
        self.playerSQN = playerSQN

    def print_board(self):
        board_symbols = [' ' if x == 0 else 'X' if x == 1 else 'O' for x in self.board]
        print("\nBoard:")
        for i in range(3):
            print(f" {board_symbols[3 * i]} | {board_symbols[3 * i + 1]} | {board_symbols[3 * i + 2]} ")
            if i < 2:
                print("---+---+---")
        print()

    def is_valid_move(self, position):
        return self.board[position] == 0

    def make_move(self, position, player):
        if self.is_valid_move(position):
            self.board[position] = player
            if self.check_winner(player):
                self.current_winner = player
            return True
        return False

    def check_winner(self, player):
        # Check all win conditions
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for condition in win_conditions:
            if all(self.board[i] == player for i in condition):
                return True
        return False

    def empty_positions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def is_full(self):
        return all(x != 0 for x in self.board)

    def player1_move(self):
        if random.random() < self.smartMovePlayer1:
            # Smart move: Try to win or block opponent
            position = self.get_smart_move()
            if position is None:
                # If no winning or blocking move, pick randomly
                position = random.choice(self.empty_positions())
        else:
            # Random move
            position = random.choice(self.empty_positions())
        self.make_move(position, 1)
        print(f"Player 1 (Smart/Random) chooses position {position + 1}")

    def get_smart_move(self):
        # Check if Player 1 can win in the next move
        for position in self.empty_positions():
            self.board[position] = 1
            if self.check_winner(1):
                self.board[position] = 0
                return position
            self.board[position] = 0

        # Check if Player 1 can block Player 2 from winning
        for position in self.empty_positions():
            self.board[position] = 2
            if self.check_winner(2):
                self.board[position] = 0
                return position
            self.board[position] = 0

        return None

    def playerSQN_move(self):
        valid_move = False
        while not valid_move:
            try:
                position = self.playerSQN.move(self.board.copy())
                if position in self.empty_positions():
                    valid_move = True
                    self.make_move(position, 2)
                else:
                    print("Invalid move, position already taken. Try again.")
            except ValueError:
                print("Invalid input, please enter a number between 1 and 9.")

    def play_game(self):
        # Player 1 is always the random or smart player, Player 2 will be SQN in the future
        self.print_board()
        player_turn = 1  # Player 1 starts
        while not self.is_full() and self.current_winner is None:
            if player_turn == 1:
                self.player1_move()
                player_turn = 2
            else:
                self.playerSQN_move()
                player_turn = 1
            self.print_board()
    
        if self.current_winner:
            winner = "Player 1 (Smart/Random)" if self.current_winner == 1 else "Player 2 (You)"
            print(f"{winner} wins!")
        else:
            print("It's a draw!")

    def get_reward(self):
        """
        Returns the reward for Player 2 (PlayerSQN):
        1 if Player 2 wins, -1 if Player 1 wins, 0 for a draw.
        """
        if self.current_winner == 2:
            return 1
        elif self.current_winner == 1:
            return -1
        else:
            return 0
