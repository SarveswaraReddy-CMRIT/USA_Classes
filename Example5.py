import numpy as np

np.random.seed(42)  # For reproducibility
rows, cols = 6, 5   # Define dimensions of the matrix
A = np.random.randint(1, 10, size=(rows, cols))  # Random integer matrix
print("Original Matrix (A):\n", A)
U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
print("\nSingular Values (Σ):\n", Sigma)

k = 2  # Number of dimensions to keep
U_k = U[:, :k]          # First k columns of U
Sigma_k = np.diag(Sigma[:k])  # Top k singular values in diagonal matrix form
VT_k = VT[:k, :]        # First k rows of V^T

A_approx = np.dot(U_k, np.dot(Sigma_k, VT_k))
print("\nReconstructed Matrix (A_approx):\n", np.round(A_approx, 2))

error = np.linalg.norm(A - A_approx, 'fro')
relative_error = error / np.linalg.norm(A, 'fro')

print("\nFrobenius Norm of Reconstruction Error:", round(error, 4))
print("Relative Error:", round(relative_error, 4))
