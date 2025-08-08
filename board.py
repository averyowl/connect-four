from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Board:
    board: tuple[str, ...] = ("", "", "", "", "", "", "")
    last_row: int = 0
    last_col: int = 0

    def get_actions(self) -> list[int]:
        return [i for i,j in enumerate(self.board) if len(j) < 6]

    def take_action(self, player: str, column: int):
        new_board: list[str] = list(self.board)
        new_board[column] = self.board[column] + player
        new_object = Board(tuple(new_board), len(new_board[column])-1, column)
        won = ""
        if(new_object.check_location(new_object.last_row, new_object.last_col)):
            won = new_object.get_space(new_object.last_row, new_object.last_col)
        return new_object, won

    def check_won(self):
        if(self.check_location(self.last_row, self.last_col)):
            return self.get_space(self.last_row, self.last_col)
        return ""
    
    def display_board(self):
        for i in reversed(range(0, 6)):
            for j in self.board:
                if(len(j)-1 >= i):
                    print(j[i], end="")
                else:
                    print(" ", end="")
            print()

    def get_space(self, row: int, col: int) -> str:
        if(len(self.board[col]) - 1 >= row):
            return self.board[col][row]
        return ""

    def check_location(self, row: int, col: int) -> bool:
        player = self.get_space(row, col)
        sequence = 0
        max_sequence = 0
        for x,y in ((1,0), (0,1), (1,1), (-1, 1)):
            sequence = 0
            for i in range(-3, 4):
                if(not self.valid_location(row+(i*x), col+(i*y))):
                    sequence = 0
                    continue
                if(self.get_space(row+(i*x), col+(i*y)) == player):
                    sequence += 1
                    if(sequence > max_sequence):
                        max_sequence = sequence
                else:
                    sequence = 0

        return max_sequence >= 4


    def valid_location(self, row: int, col: int) -> bool:
        if(col < 0 or col >= 7):
            return False
        if(row < 0 or row > (len(self.board[col]) - 1)):
            return False
        return True
