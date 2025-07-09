import random

class NQueensState:
    def __init__(self, n=8, initial_board=None):
        """
        Constructor for the NQueensState class.
        If no board is provided, generates a random one.

        :param n: Size of the board (number of queens)
        :param initial_board: Optional - a list of queen positions
        """
        self.n = n 

        if initial_board is None:
            #If no board is provided, generate a random board
            #Each element in the list represents the column of the queen in that row
            self.board = [random.randint(0, n - 1) for _ in range(n)]
        else:
            self.board = initial_board

    def __str__(self):
        """
        Converts the object to a string when you print it.

        :return: A string representation of the board
        """
        return str(self.board)

    def conflicts(self):
        """
        Calculates the number of conflicting queen pairs on the board.

        :return: Total number of attacking pairs
        """
        count = 0  #Start conflict count from zero

        #Loop through every unique pair of queens
        for i in range(self.n):
            for j in range(i + 1, self.n):  #Avoid checking same and already checked pairs
                #Check if two queens are in the same column
                if self.board[i] == self.board[j]:
                    count += 1  # They are attacking each other in the same column

                #Check if they are on the same diagonal
                #Diagonal if the difference in row equals difference in column
                elif abs(self.board[i] - self.board[j]) == abs(i - j):
                    count += 1 

        return count

    def get_neighbours(self):
        """
        Generates all valid neighbouring board states by moving one queen
        to another column in her row.

        :return: List of new NQueensState objects
        """
        neighbours = []  # Start with an empty list of neighbours

        # Go through every row (each row has exactly one queen)
        for row in range(self.n):
            current_col = self.board[row]  #Get current column of the queen in this row

            #Try every other column in this row
            for col in range(self.n):
                if col == current_col:
                    continue  #Skip if it's the same column — not a valid move

                #Make a copy of the current board
                new_board = self.board.copy()

                #Move the queen in the current row to a new column
                new_board[row] = col

                #Create a new state with this new board and add to neighbours
                neighbours.append(NQueensState(self.n, new_board))

        return neighbours  # eturn the full list of neighbouring states
