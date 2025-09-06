#1d_gradient_descent.py

#This script implements 1-D Gradient Descent (GD) in a purely procedural style.
#Only the fixed-step variant is included; `eta` is the learning rate.
#Per-step history can be saved to a CSV file, and a final summary is written to a text file.

from typing import List, Dict, Tuple
import csv

#---------------------------------
#Hard-coded objective and gradient
#---------------------------------
#f(x)=(x-3)^2 with derivative f'(x)=2(x-3).

def function_to_minimize(x: float) -> float:
    #Return f(x)
    return (x - 3.0)**2


def derivative_function(x: float) -> float:
    #Return f'(x)
    return 2.0*(x - 3.0)


#---------------------------------
#Utility: write history to CSV
#---------------------------------

def write_history_to_csv(csv_path: str, history: List[Dict[str, float]]) -> None:
    #Write list-of-dicts history to a CSV file
    if not history:
        return
    fieldnames=list(history[0].keys())
    with open(csv_path, mode="w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


#---------------------------------
#Algorithm — Fixed-step GD (eta)
#---------------------------------

def gd_1d_fixed(
    starting_x: float,
    eta: float=0.2,
    gradient_tolerance: float=1e-8,
    step_tolerance: float=1e-8,
    max_iterations: int=10_000,
    keep_history: bool=True,
    f=function_to_minimize,
    df=derivative_function,
) -> Tuple[float, float, int, str, List[Dict[str, float]]]:
    #Run 1-D gradient descent with a fixed learning rate eta.
    #Returns: best_x, best_f, iteration_count, convergence_reason, history

    current_x=float(starting_x)
    history: List[Dict[str, float]]=[]
    convergence_reason: str=""

    for iteration_count in range(max_iterations):
        g=df(current_x)

        #Stationarity check: near-flat slope
        if abs(g)<=gradient_tolerance:
            convergence_reason="small_gradient"
            break

        next_x=current_x - eta*g
        step_size=abs(next_x - current_x)

        #Small-step check: negligible movement
        if step_size<=step_tolerance:
            current_x=next_x
            convergence_reason="small_step"
            break

        if keep_history:
            history.append({
                "iteration": float(iteration_count),
                "x": current_x,
                "f(x)": f(current_x),
                "abs_grad": abs(g),
                "step_size": step_size,
                "eta": eta,
            })

        current_x=next_x
    else:
        convergence_reason="iteration_cap"

    best_x=current_x
    best_f=f(best_x)
    return best_x, best_f, iteration_count, convergence_reason, history


#-----------------
#Example execution
#-----------------
if __name__=="__main__":
    #Settings
    starting_x=5.0
    eta=0.2
    eps_g=1e-8
    eps_x=1e-8
    T_max=10_000

    #Run fixed-step GD
    best_x, best_f, iters, reason, hist=gd_1d_fixed(
        starting_x=starting_x,
        eta=eta,
        gradient_tolerance=eps_g,
        step_tolerance=eps_x,
        max_iterations=T_max,
        keep_history=True,
    )

    #Write history to CSV file
    write_history_to_csv("gd_1d_history_fixed.csv", hist)

    #Write summary to a text file (no screen prints)
    with open("gd_1d_results.txt", "w") as outf:
        outf.write("Fixed-step GD:")
        outf.write(f"best_x={best_x}")
        outf.write(f"best_f={best_f}")
        outf.write(f"iterations={iters}")
        outf.write(f"reason={reason}")
