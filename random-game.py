import random
from board import Board
from game import Game

def choose_random(actions: list[int], state: Board):
    return random.choice(actions)

game = Game(choose_random, choose_random)

game.run()

print(game.winner)
game.state.display_board()


