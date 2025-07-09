from n_queens_state import NQueensState
from hill_climbing_solver import HillClimbingSolver

def main():
    # Create an initial board of size 8 
    initial_state = NQueensState(n=8)

    # Create the solver
    solver = HillClimbingSolver(n=8, initial_state=initial_state)

    # Solve the problem
    solution = solver.solve()
    print("Solved Board:", solution.board)

if __name__ == "__main__":
    main()
