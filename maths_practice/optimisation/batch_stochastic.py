#Author: Anoop K. Saxena
#Purpose: Batch Gradient Descent + Stochastic Gradient Descent using autograd (3 params: θ0, θ1, θ2)

import autograd.numpy as np
from autograd import grad
rng = np.random.RandomState(42)  # reproducibility

#-----------------------------
# Toy data (m×2 features)
#-----------------------------
X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0],
    [4.0, 3.0],
    [5.0, 5.0]
])
y = np.array([5.0, 5.0, 11.0, 11.0, 16.0])

def add_bias(X):
    # Returns design matrix with bias column: shape (m, 3) → [1, x1, x2]
    m = X.shape[0]
    return np.column_stack([np.ones(m), X])

Xb = add_bias(X)   # (m,3)
m = Xb.shape[0]

#-----------------------------
# Cost: J(θ) = (1/2m) * ||Xθ - y||^2
#-----------------------------
def cost(theta):
    preds = Xb @ theta                # (m,)
    residuals = preds - y             # (m,)
    return 0.5 * np.mean(residuals**2)

# Autograd gradient: ∇J(θ) = (1/m) X^T (Xθ - y)
grad_cost = grad(cost)

#-----------------------------
# Batch Gradient Descent
#-----------------------------
def gradient_descent(eta=0.05, max_iters=200, eps=1e-8, verbose_every=10):
    theta = np.zeros(3)  # [θ0, θ1, θ2]
    for t in range(max_iters):
        c = cost(theta)
        if (t % verbose_every) == 0:
            print(f"[BGD] iter={t:4d}  cost={c:.8f}  theta={theta}")
        g = grad_cost(theta)
        new_theta = theta - eta * g
        if np.linalg.norm(new_theta - theta) < eps:
            theta = new_theta
            print(f"[BGD] converged at iter={t}, cost={cost(theta):.8f}, theta={theta}")
            return theta
        theta = new_theta
    print(f"[BGD] done. final cost={cost(theta):.8f}, theta={theta}")
    return theta

#-----------------------------
# Per-sample loss: ℓ_i(θ) = (1/2) * (x_i^T θ - y_i)^2
#-----------------------------
def make_sample_loss(x_i, y_i):
    # Returns a scalar function ℓ_i(θ) for a fixed (x_i, y_i)
    def loss_i(theta):
        return 0.5 * (x_i.dot(theta) - y_i) ** 2
    return loss_i

#-----------------------------
# Stochastic Gradient Descent (SGD)
#   - Shuffles data each epoch
#   - Updates θ per single example
#   - Tracks and prints full-batch cost per epoch
#-----------------------------
def stochastic_gradient_descent(eta=0.05, epochs=50, eps=1e-8, verbose_every=1):
    theta = np.zeros(3)
    indices = np.arange(m)

    # Pre-build gradient fns for each sample (so autograd comp graph is simple per-sample)
    sample_grads = []
    for i in range(m):
        loss_i = make_sample_loss(Xb[i], y[i])
        sample_grads.append(grad(loss_i))

    prev_theta = theta.copy()
    for ep in range(1, epochs + 1):
        rng.shuffle(indices)

        # One pass over data (stochastic updates)
        for i in indices:
            g_i = sample_grads[i](theta)   # ∇ℓ_i(θ)
            theta = theta - eta * g_i

        # Check/report at epoch boundary using full-batch cost
        c = cost(theta)
        if (ep % verbose_every) == 0:
            print(f"[SGD]  epoch={ep:4d}  cost={c:.8f}  theta={theta}")

        # Simple stopping on parameter movement per epoch
        if np.linalg.norm(theta - prev_theta) < eps:
            print(f"[SGD]  converged at epoch={ep}, cost={c:.8f}, theta={theta}")
            return theta
        prev_theta = theta.copy()

    print(f"[SGD]  done. final cost={cost(theta):.8f}, theta={theta}")
    return theta

#-----------------------------
# Main
#-----------------------------
if __name__ == "__main__":
    print("=== Batch Gradient Descent ===")
    theta_bgd = gradient_descent(eta=0.05, max_iters=500, verbose_every=20)

    print("\n=== Stochastic Gradient Descent ===")
    theta_sgd = stochastic_gradient_descent(eta=0.05, epochs=200, verbose_every=10)

    print("\nBGD  θ* =", theta_bgd)
    print("SGD  θ* =", theta_sgd)
