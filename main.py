# Linearization Methods: Newton-Raphson, Modified Newton-Raphson, Line Search, BFGS
# Author: Fernanda Fontenele (fernandaffontenele@gmail.com)
# Reference: Here I follow the the theory by Wriggers in the "Nonlinear Finite Element Methods" book.
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Define here the function you want to solve for (we are trying to solve G(v) = R(v) - y * P = 0 ),
# where R(v) is the function we are trying to solve for, y is the loading parameter and P is the load term.
def R_function(v, x):  # Here, solving for the following
    R = np.zeros((2, 1))  # R1 = 0.2 * v1^3 -x * v2^2 + 6 * v1  and  R2 = v2 - v1
    R[0], R[1] = 0.2 * (v[0] ** 3) - x * (v[1] ** 2) + 6 * v[0], v[1] - v[0]
    return R

def KT_matrix(v, x):  # Define the tangent matrix (which is the linearization of G)
    KT = np.zeros((2, 2))
    KT[0, 0], KT[0, 1], KT[1, 0], KT[1, 1] = 0.6 * (v[0] ** 2) + 6, -2 * x * v[1], -1, 1
    return KT

def analytical_sln(x, y):  # Define here the analytical solution if desired (for comparison to numeircal calculation)
    coeff = [0.2, -x, 6, -y]
    soln = np.roots(coeff)
    v = soln[np.isreal(soln)]
    R = 0.2 * (v ** 3) - x * (v ** 2) + 6 * v
    return v, R

if __name__ == '__main__':
    # USER INPUT PARAMETERS - Initialize the program -------------------------------------------------------------------
    Load_inc = np.arange(0, 10, 0.25)  # Load parameter: it is a scaling factor for the load P
    P = np.array([[1], [0]])  # Load term
    v = np.array([[1], [1]])  # Solution first guesses
    x = 2.1  # Additional parameter of equation to be solved
    epsilon, max_iter = 1e-4, 15  # Convergence parameters: convergence criterion and maximum # of iterations
    Method = "BFGS"  # Choose here the optimization method: "NR" for Newton-Raphson, "MNR" for modified
    # Newton-Raphson, "MNR_LS" for modified Newton-Raphson with line search or "BFGS" for BFGS method. -----------------

    # Initialize vectors
    v_ana, R_ana = np.zeros((len(Load_inc), 1)), np.zeros((len(Load_inc), 1))
    R1_num, v1_num, niter = np.zeros((len(Load_inc), 1)), np.zeros((len(Load_inc), 1)), np.zeros((len(Load_inc), 1))

    tic = time.perf_counter()
    for i in range(len(Load_inc)):  # Loop over all load parameters
        y = Load_inc[i]  # take the current load parameter
        G = R_function(v, x) - y * P  # Find the first value of G
        tol = epsilon * np.linalg.norm(G)  # Convergence: comparison between norm of G in step k+1 to the norm of G in step 0
        v_ana[i], R_ana[i] = analytical_sln(x, y)  # Analytical calculation

        iteration = 0
        if Method == "NR":  # Method Newton-Raphson
            while np.linalg.norm(G) > tol and iteration <= max_iter:  # (Convergence is based on Euclidean Norm)
                iteration = 1 + iteration
                v = v - np.linalg.inv(KT_matrix(v, x)) @ G  # Solve the linear system and find the updated solution
                G = R_function(v, x) - y * P  # Find the new G, calculated from the updated v

        if Method == "MNR":  # Method Modified Newton-Raphson
            KT_inv = np.linalg.inv(KT_matrix(v, x))  # Tangent matrix is only calculated and inverted once in MNR
            while np.linalg.norm(G) > tol and iteration <= max_iter:
                iteration = 1 + iteration
                v = v - KT_inv @ G  # Solve the linear system and find the updated solution
                G = R_function(v, x) - y * P  # Find the new G, calculated from the updated v

        if Method == "MNR_LS":  # Method Modified Newton-Raphson with Line Search
            KT_inv, g, alpha = np.linalg.inv(KT_matrix(v, x)), np.zeros((2, 1)), np.zeros((2, 1))
            while np.linalg.norm(G) > tol and iteration <= max_iter:
                iteration = 1 + iteration
                Dv = - KT_inv @ G
                for a in range(2):
                    alpha[a] = a
                    G = R_function((v + a * Dv), x) - y * P  # This gives us G(v_i + alpha_i Dv_i+1, y)
                    g[a] = np.transpose(Dv) @ G  # This gives us g(alpha_i) = Dv_i+1^T G(v_i + alpha_i Dv_i+1, y)
                if g[0] * g[1] < 0:  # Check if the root of g is in the interval 0<alpha<1
                    g0, alpha_iter = g[0], 0
                    while abs(g[1]) > 0.5 * abs(g0) and alpha_iter < 5:  # Convergence for line search
                        # Compute alpha_i for which g(alpha_i) = 0 using line search
                        new_alpha = alpha[1] - g[1] * (alpha[1] - alpha[0]) / (g[1] - g[0])
                        alpha[0], alpha[1] = alpha[1], new_alpha  # Update alpha_k-1 and alpha_k for next iteration
                        # Now calculate G for the new alpha
                        G = R_function((v + alpha[1] * Dv), x) - y * P
                        # Then update the g function we are trying to minimize through the alpha parameter
                        g[0] = g[1]  # Update g_k-1 for next alpha iteration
                        g[1] = np.transpose(Dv) @ G  # Find g_k for the next alpha iteration and to check if converged
                        alpha_iter = alpha_iter + 1
                else:
                    alpha[1] = 1  # Goes back to normal MNR
                v = v + alpha[1] * Dv  # Solve the linear system and find the updated solution
                G = R_function(v, x) - y * P  # Find the new G, calculated from the updated v

        if Method == "BFGS":  # Method BFGS
            KT_inv = np.linalg.inv(KT_matrix(v, x))
            while np.linalg.norm(G) > tol and iteration <= max_iter:
                iteration = 1 + iteration
                new_v = v - KT_inv @ G  # Solve the linear system and find the updated solution
                new_G = R_function(new_v, x) - y * P  # Find the new G, calculated from the updated v
                H, w, g = KT_inv, new_v - v,  - new_G - G  # Calculate BFGS parameters
                a = (1 / (np.transpose(g) @ w)) * w
                b = - g + ((np.transpose(w) @ g / (np.transpose(w) @ (-G))) ** (1 / 2)) * G
                KT_inv = (np.eye(2) + a @ np.transpose(b)) @ H @ (np.eye(2) + b @ np.transpose(a))
                v, G = new_v, new_G  # Update

        R1_num[i], v1_num[i], niter[i] = R_function(v, x)[0], v[0], iteration

    toc = time.perf_counter()
    print(f"1. The solution is v = {v}  and running time t =  {toc - tic:0.4f} s")

    plot1 = plt.figure(1)  # Plot the numerical and analytical solution
    plt.plot(v_ana, R_ana, 'k')
    plt.plot(v1_num, R1_num, '--')
    plot2 = plt.figure(2)
    plt.plot(range(len(Load_inc)), niter, 'o')
    ax = plot2.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()