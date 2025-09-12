#Author: Anoop K. Saxena
import os
import numpy as np

def get_matrix():
    #Default hardcoded matrix (square example)
    return np.array([
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0]
    ], dtype=float)

def fix_row(A: np.ndarray, i: int, eps: float = 1e-12) -> bool:
    #Eliminate contributions from previously-fixed rows 0..i-1
    for j in range(i):
        A[i] -= A[i, j]*A[j]
    #If pivot near zero, try partial pivoting
    if abs(A[i, i]) < eps:
        swap = -1
        for r in range(i+1, A.shape[0]):
            if abs(A[r, i]) >= eps:
                swap = r
                break
        if swap == -1:
            return False
        A[[i, swap]] = A[[swap, i]]
        for j in range(i):
            A[i] -= A[i, j]*A[j]
        if abs(A[i, i]) < eps:
            return False
    A[i] /= A[i, i]
    return True

def is_singular_square(A: np.ndarray, eps: float = 1e-12) -> bool:
    B = np.array(A, dtype=float, copy=True)
    n = B.shape[0]
    for i in range(n):
        if not fix_row(B, i, eps=eps):
            return True
    return False

def is_rank_deficient(A: np.ndarray, eps: float = 1e-12) -> bool:
    r = np.linalg.matrix_rank(A, tol=eps)
    return r < min(A.shape[0], A.shape[1])

def main():
    A = get_matrix()
    m, n = A.shape
    if m == n:
        singular = is_singular_square(A)
        verdict = "Singular" if singular else "Not singular"
        details = "method=Gaussian elimination with partial pivoting"
    else:
        singular = is_rank_deficient(A)
        verdict = "Rank-deficient (rectangular)" if singular else "Full rank (rectangular)"
        details = f"rank={np.linalg.matrix_rank(A)} min(m,n)={min(m,n)}"

    msg = []
    msg.append("Matrix:")
    msg.append(str(A))
    msg.append(f"shape={A.shape}")
    msg.append(details)
    msg.append(f"Result: {verdict}")

    out_text = "\n".join(msg)
    print(out_text)

    #Also save output to file in same directory
    out_path = os.path.join(os.path.dirname(__file__), "singular_output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text+"\n")

if __name__ == "__main__":
    main()
