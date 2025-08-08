from board import Board
from typing import Callable

class Game:
    state: Board = Board()
    winner: str = ""

    def __init__(self, one_choice: Callable[[list[int], Board, str, str], int], two_choice: Callable[[list[int], Board, str, str], int]) -> None:
        self.one_choice: Callable[[list[int], Board, str, str], int] = one_choice
        self.two_choice: Callable[[list[int], Board, str, str], int] = two_choice

    def run(self):
        while(len(self.state.get_actions()) > 0):
            self.state,won = self.state.take_action('X', self.one_choice(self.state.get_actions(), self.state, 'X', 'O'))
            if(won != ""):
                self.winner = "X"
                break
            self.state,won = self.state.take_action('O', self.two_choice(self.state.get_actions(), self.state, 'O', 'X'))
            if(won != ""):
                self.winner = "O"
                break


