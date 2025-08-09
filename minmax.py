# this program is a minmax algorithm implementation, that uses the connect-four game and board in this folder

# Josh Davis
# 2025-08-06
from board import Board
from game import Game


# Class: MinMax
# This class implements the MinMax algorithm for Connect Four.
# It evaluates the game state and finds the best move for the player
# it also tracks various metrics related to the MinMax algorithm's performance so we can do some comparisons
class MinMax:
    def __init__(self, player='X', opponent='O'):
        self.player = player
        self.opponent = opponent
        #Metrics tracking
        self.metrics = {
            'total_minimax_calls': 0,
            'moves_made': 0,
            'average_calls_per_move': 0,
            'max_calls_in_single_move': 0,
            'min_calls_in_single_move': float('inf'),
            'terminal_nodes_reached': 0,
            'evaluation_calls': 0,
            'wins_found': 0,
            'losses_found': 0,
            'draws_found': 0,
            'move_times': [],
            'search_depths': []
        }
        self.current_move_calls = 0

    #Function: is_terminal
    #Input: self, board
    #Output: bool
    #Description: Checks if the game state is terminal meaning there is no more valid moves or somebody won
    def is_terminal(self, board):
        return len(board.get_actions()) == 0 or board.check_won() != ""
    
    #Function: get_valid_moves
    #Input: self, board
    #Output: list[int]
    #Description: Returns a list of valid moves (columns) for the current board state
    def get_valid_moves(self, board):
        return board.get_actions()
    
    #Function: make_move
    #Input: board, move, player
    #Output: Board
    #Description: Takes a move on the board and returns the new board state this is the meat of the simulation
    def make_move(self, board, move, player):
        new_board, _ = board.take_action(player, move)
        return new_board
    
    #Function: evaluate
    #Input: board
    #Output: int
    #Description: Evaluates the board state and returns a score based on the current player's perspective
    def evaluate(self, board):
        self.metrics['evaluation_calls'] += 1
        
        winner = board.check_won()
        if winner == self.player:
            self.metrics['wins_found'] += 1
            return 1000  #hih positive score for winning
        elif winner == self.opponent:
            self.metrics['losses_found'] += 1
            return -1000  #high negative score for losing
        elif len(board.get_actions()) == 0:
            self.metrics['draws_found'] += 1
            return 0  #Draw
        else:
            #heuristic evaluation for non-terminal positions
            return self._evaluate_position(board)
    
    #Function: _evaluate_position
    #Input: board
    #Output: int
    #Description: Evaluates the board position and returns a score based on potential winning sequences, 
    # we could put this in the above evaluate, but then that would be a beefy function
    def _evaluate_position(self, board):
        score = 0
        
        #Evaluate all possible 4-in-a-row positions
        for col in range(7):
            for row in range(6):
                #Check horizontal, vertical, and both diagonals
                for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    score += self._evaluate_window(board, row, col, direction)
        
        return score
    
    #Function: _evaluate_window
    #Input: board, row, col, direction
    #Output: int
    #Description: Evaluates a window of 4 pieces in the given direction and returns a score based on the pieces in that window.
    # This is a helper function for the above _evaluate_position
    def _evaluate_window(self, board, row, col, direction):
        dx, dy = direction
        window = []
        
        #Collect 4 pieces in the given direction
        for i in range(4):
            new_row = row + i * dx
            new_col = col + i * dy
            if board.valid_location(new_row, new_col):
                window.append(board.get_space(new_row, new_col))
            else:
                return 0  #Invalid window, oopsie daisy
        
        return self._score_window(window)
    
    #Function: _score_window
    #Input: window
    #Output: int
    #Description: Scores a window of 4 pieces based on the player's and opponent's pieces
    # This is another helper function for the above _evaluate_position
    def _score_window(self, window):
        score = 0
        player_count = window.count(self.player)
        opponent_count = window.count(self.opponent)
        empty_count = window.count("")
        
        #these scores can be adjusted for different heuristics if we want
        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 10
        elif player_count == 2 and empty_count == 2:
            score += 2
        
        if opponent_count == 3 and empty_count == 1:
            score -= 80  #Block opponent
        
        return score

    #Function: minmax
    #Input: board, depth, maximizing_player, current_player
    #Output: int
    #Description: Implements the MinMax algorithm recursively to find the best move for the current player
    def minmax(self, board, depth, maximizing_player, current_player):
        self.metrics['total_minimax_calls'] += 1
        self.current_move_calls += 1
        
        if depth == 0 or self.is_terminal(board):
            self.metrics['terminal_nodes_reached'] += 1
            return self.evaluate(board)

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_valid_moves(board):
                new_board = self.make_move(board, move, current_player)
                next_player = self.opponent if current_player == self.player else self.player
                eval = self.minmax(new_board, depth - 1, False, next_player)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_valid_moves(board):
                new_board = self.make_move(board, move, current_player)
                next_player = self.opponent if current_player == self.player else self.player
                eval = self.minmax(new_board, depth - 1, True, next_player)
                min_eval = min(min_eval, eval)
            return min_eval

    #Function: find_best_move
    #Input: board, depth
    #Output: int
    #Description: Finds the best move for the current player using the MinMax algorithm
    # This function iterates through all valid moves, simulates them, and uses MinMax
    def find_best_move(self, board, depth):
        import time
        start_time = time.time()
        self.current_move_calls = 0
        
        best_move = None
        best_value = float('-inf')

        for move in self.get_valid_moves(board):
            new_board = self.make_move(board, move, self.player)
            eval = self.minmax(new_board, depth - 1, False, self.opponent)
            if eval > best_value:
                best_value = eval
                best_move = move

        #update metrics
        move_time = time.time() - start_time
        self.metrics['move_times'].append(move_time)
        self.metrics['search_depths'].append(depth)
        self.metrics['moves_made'] += 1
        
        #Update call statistics
        if self.current_move_calls > self.metrics['max_calls_in_single_move']:
            self.metrics['max_calls_in_single_move'] = self.current_move_calls
        if self.current_move_calls < self.metrics['min_calls_in_single_move']:
            self.metrics['min_calls_in_single_move'] = self.current_move_calls
            
        #calculate average
        self.metrics['average_calls_per_move'] = (
            self.metrics['total_minimax_calls'] / self.metrics['moves_made']
        )
        
        return best_move

    #Function: get_metrics_summary
    #Input: None
    #Output: str
    #Description: Returns a summary of the MinMax agent's metrics in a pretty string for your terminal ok?
    def get_metrics_summary(self):
        if self.metrics['moves_made'] == 0:
            return "No moves made yet - no metrics to display"
        
        avg_time = sum(self.metrics['move_times']) / len(self.metrics['move_times'])
        total_time = sum(self.metrics['move_times'])
        
        summary = f"""
=== MinMax Agent Metrics ===
Total Moves Made: {self.metrics['moves_made']}
Total Minimax Calls: {self.metrics['total_minimax_calls']}
Average Calls per Move: {self.metrics['average_calls_per_move']:.1f}
Max Calls in Single Move: {self.metrics['max_calls_in_single_move']}
Min Calls in Single Move: {self.metrics['min_calls_in_single_move']}

Terminal Nodes Reached: {self.metrics['terminal_nodes_reached']}
Evaluation Function Calls: {self.metrics['evaluation_calls']}

Outcomes Found:
  - Wins: {self.metrics['wins_found']}
  - Losses: {self.metrics['losses_found']}
  - Draws: {self.metrics['draws_found']}

Timing:
  - Average Time per Move: {avg_time:.3f}s
  - Total Thinking Time: {total_time:.3f}s
  - Fastest Move: {min(self.metrics['move_times']):.3f}s
  - Slowest Move: {max(self.metrics['move_times']):.3f}s

Search Depths Used: {set(self.metrics['search_depths'])}
        """
        return summary.strip()
    
    #Function: print_metrics
    #Input: None
    #Output: None
    #Description: Prints the metrics summary to the console, duh
    def print_metrics(self):
        print(self.get_metrics_summary())
    
    #Function: reset_metrics
    #Input: None
    #Output: None
    #Description: Resets the metrics to their initial state
    # This is useful if we want to run multiple games and track metrics separately
    def reset_metrics(self):
        self.metrics = {
            'total_minimax_calls': 0,
            'moves_made': 0,
            'average_calls_per_move': 0,
            'max_calls_in_single_move': 0,
            'min_calls_in_single_move': float('inf'),
            'terminal_nodes_reached': 0,
            'evaluation_calls': 0,
            'wins_found': 0,
            'losses_found': 0,
            'draws_found': 0,
            'move_times': [],
            'search_depths': []
        }
        self.current_move_calls = 0

    #Function: get_live_stats
    #Input: None
    #Output: dict
    #Description: Returns a dictionary of live stats for the MinMax agent, useful for monitoring
    def get_live_stats(self):
        return {
            'current_move_calls': self.current_move_calls,
            'total_calls': self.metrics['total_minimax_calls'],
            'moves_made': self.metrics['moves_made']
        }

    #Function: get_choice_function
    #Input: depth=4
    #Output: Callable
    #Description: Returns a callable function that can be used as a choice function for the Game
    # This function allows us to use the MinMax agent in the Game class without modifying its interface
    # It takes the depth as an argument to control how deep the MinMax algorithm will search
    # This is useful for tuning the performance of the MinMax agent. I was pretty pleased with this
    def get_choice_function(self, depth=4):
        def choice_function(actions, state, my_symbol, opponent_symbol):
            # Update player symbols for this game
            self.player = my_symbol
            self.opponent = opponent_symbol
            return self.find_best_move(state, depth)
        return choice_function


if __name__ == "__main__":
    #create a MinMax AI
    ai = MinMax('X', 'O')
    
    #create a choice function for the AI
    ai_choice = ai.get_choice_function(depth=4)  #Reduced depth for faster demonstration, we can deepen this without too much trouble
    
    #random choice function for opponent
    import random
    def random_choice(actions, state, my_symbol, opponent_symbol):
        return random.choice(actions)
    
    print("Starting Connect Four: AI vs Random Player")
    print("AI is playing as 'X', Random player as 'O'")
    print("=" * 50)
    
    #create and run a game with the AI vs Random moves
    game = Game(ai_choice, random_choice)
    game.run()
    
    print(f"\nGame Over! Winner: {game.winner if game.winner else 'Draw'}")
    print("\nFinal Board:")
    game.state.display_board()
    
    print("\n" + "=" * 50)
    ai.print_metrics()
    
    #Optional: Show some additional analysis
    if ai.metrics['moves_made'] > 0:
        efficiency = ai.metrics['terminal_nodes_reached'] / ai.metrics['total_minimax_calls'] * 100
        print(f"\nSearch Efficiency: {efficiency:.1f}% of calls reached terminal nodes")
        
        if ai.metrics['move_times']:
            print(f"Thinking Speed: {ai.metrics['total_minimax_calls'] / sum(ai.metrics['move_times']):.0f} calls/second")