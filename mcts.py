import random
from board import Board
from game import Game

def ask(actions: list[int], state: Board, my_symbol: str, opponent_symbol: str) -> int:
    state.display_board()
    print("1234567")
    choice = input("choose a location: ")
    return int(choice) - 1

def mcts(actions: list[int], state: Board, my_symbol: str, opponent_symbol: str) -> int:
    best: dict[int, int] = {}
    for i in actions:
        best[i] = 0

    for i in range(0, 500):
        states: list[Board] = []
        first_action = random.choice(actions)
        new_state, won = state.take_action(my_symbol, first_action)
        if(won == my_symbol):
            best[first_action] += 1
            continue
        states.append(new_state)
        my_turn = False
        while(len(new_state.get_actions()) > 0):
            new_state,won = new_state.take_action(my_symbol if my_turn else opponent_symbol, random.choice(new_state.get_actions()))
            if(won == my_symbol):
                best[first_action] += 1
                break
            if(won == opponent_symbol):
                best[first_action] -= 1
                break
            states.append(new_state)
            my_turn = not my_turn
    return int(max(best, key=best.get))

def choose_random(actions: list[int], state: Board, my_symbol: str, opponent_symbol: str):
    return random.choice(actions)

if __name__ == "__main__":

    winner = {"X": 0, "O": 0, "": 0}
    for i in range(25):
        game = Game(choose_random, mcts)
        game.run()
        print(game.winner)
        game.state.display_board()
        winner[game.winner] += 1

    print(winner)

