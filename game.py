from board import Board
from typing import Callable

class Game:
    state: Board
    winner: str

    def __init__(
        self,
        one_choice: Callable[[list[int], Board, str, str], int],
        two_choice: Callable[[list[int], Board, str, str], int],
    ) -> None:
        # Choice functions follow signature:
        #   (actions, state, my_symbol, opponent_symbol) -> column (int)
        self.one_choice = one_choice
        self.two_choice = two_choice
        self.state = Board()
        self.winner = ""  # "X", "O", or "D" for draw

    def run(self) -> str:
        """Run a game until win or draw.
        
        Returns:
            str: "X" if player X wins, "O" if player O wins, "D" if draw.
        """
        # Alternate X then O until someone wins or the board is full.
        while len(self.state.get_actions()) > 0:
            # X moves
            self.state, won = self.state.take_action(
                'X',
                self.one_choice(self.state.get_actions(), self.state, 'X', 'O')
            )
            if won != "":
                self.winner = 'X'
                break

            # O moves
            self.state, won = self.state.take_action(
                'O',
                self.two_choice(self.state.get_actions(), self.state, 'O', 'X')
            )
            if won != "":
                self.winner = 'O'
                break

        # If no actions left and no winner, it's a draw.
        if self.winner == "" and len(self.state.get_actions()) == 0:
            self.winner = 'D'

        return self.winner
