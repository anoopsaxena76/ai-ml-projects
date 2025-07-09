from n_queens_state import NQueensState

class HillClimbingSolver:
    def __init__(self, n, initial_state):
        """
        Constructor for HillClimbingSolver.
        Stores initial board and checks if it's already a solution.

        :param n: Board size
        :param initial_state: A NQueensState object to start with
        """
        self.n = n 
        self.current_state = initial_state 

        #Check if the starting board is already a goal state
        if self.is_goal(self.current_state):
            print("Initial state is already a solution:")
            print(self.current_state)
            exit() 

    def is_goal(self, state):
        """
        Checks if the given state is a goal (no attacking queens)

        :param state: A NQueensState object
        :return: True if it has zero conflicts, False otherwise
        """
        return state.conflicts() == 0

    def get_best_neighbour(self, state):
        """
        Finds the neighbour with the fewest conflicts.

        :param state: A NQueensState object
        :return: The best neighbour state found
        """
        best_neighbour = None  #Placeholder for best state
        lowest_conflict = float('inf')  #Start with infinite conflict to ensure replacement

        neighbours = state.get_neighbours()  #Get all possible neighbouring boards

        for neighbour in neighbours:
            conflict = neighbour.conflicts()  #Check how many conflicts this neighbour has

            if conflict < lowest_conflict:
                #Found a better neighbour with fewer conflicts
                lowest_conflict = conflict
                best_neighbour = neighbour

        return best_neighbour  #Return the best one found

    def solve(self):
        """
        Main solving loop using hill climbing with restarts.

        :return: A NQueensState that is a goal (solution)
        """
        steps = 0  #how many steps we've taken

        while not self.is_goal(self.current_state):
            best_neighbour = self.get_best_neighbour(self.current_state)
            steps += 1  #Increase step count every time we evaluate a new neighbour

            #If the best neighbour is not better than the current one, it's a local minimum
            if best_neighbour.conflicts() >= self.current_state.conflicts():
                print(f"Restarting at step {steps} with a new random board")
                self.current_state = NQueensState(self.n)  #Restart with a new random board
            else:
                self.current_state = best_neighbour  #Move to the better state

        #We have reached a goal state
        print("Goal reached!")
        print("Final board:", self.current_state)
        print("Conflicts:", self.current_state.conflicts())
        return self.current_state
