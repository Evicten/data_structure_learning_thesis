import numpy as np
from scipy.integrate import quad
from scipy.linalg import sqrtm
import sys
from mpi4py import MPI
from typing import Callable
from matplotlib import pyplot as plt
from scipy.integrate import dblquad



def beta_tilde(beta_v):
    return beta_v/(1+beta_v+np.sqrt(1+beta_v))

def gamma_matrix(beta_v):
    mat = np.zeros((2,2))
    mat[1,1] = beta_v

    return mat

def beta_vec(nu, lambda_, beta):
    beta_u, beta_v = beta
    return np.array([np.sqrt(beta_u)*lambda_, np.sqrt(beta_v)/(1-beta_tilde(beta_v))*nu])
   
def int_1(a,b,c,d):
    return 1/np.sqrt(a+1)*np.exp((b**2+d**2)/(2*(a+1)))*np.cosh(c+b*d/(a+1))

def int_2(a,b,c,d):
    a = a+1
    return 1/(np.sqrt(a)*a)*np.exp((b**2+d**2)/(2*a))*(b*np.cosh(c+b*d/a) + d*np.sinh(c+b*d/a))

def int_3(a,b,c,d):
    a = a+1
    return 1/np.sqrt(a)*np.exp((b**2+d**2)/(2*a))*np.sinh(c+b*d/a)

def numerator_vec(omega, V, beta, a, b, c, d, eps=1e-6):
    omega1, omega2 = omega
    beta_u, beta_v = beta
    beta_tilde_v = beta_tilde(beta_v)
    gamma = gamma_matrix(beta_v)
    V_inv = np.linalg.inv(V+np.eye(2)*eps)
    M = np.linalg.inv(V_inv+gamma+np.eye(2)*eps)
    N = V_inv @ M
    vec1 = np.sqrt(beta_u)*int_2(a,b,c,d)
    vec2 = np.sqrt(beta_v)/(1-beta_tilde_v)*int_3(a,b,c,d)
    vec = np.array([vec1, vec2])
    return N @ vec - N @ gamma  @ omega * int_1(a,b,c,d)


def q_func(q_hat):
    return q_hat @ np.linalg.inv(q_hat + np.eye(2))

def q_hat_func_MCMC(alpha, beta, q, samples, eps=1e-6):
    expectation_xi = np.zeros((2,2))
    beta_u, beta_v = beta
    sqrt_q = sqrtm(q)
    V = np.eye(2) - q
    gamma = gamma_matrix(beta_v)
    V_inv = np.linalg.inv(V+np.eye(2)*eps)
    M = np.linalg.inv(V_inv+gamma+np.eye(2)*eps)
    N = V_inv @ M
    beta_tilde_v = beta_tilde(beta_v)
    a = beta_u*(1-M[0,0])
    d = np.sqrt(beta_u*beta_v)/(1-beta_tilde_v)*M[0,1]

    for _ in range(samples):
        Xi = np.random.normal(0,1, 2)
        omega = sqrt_q @ Xi
        omega1, omega2 = omega
        b = np.sqrt(beta_u)*(omega1*N[0,0]+omega2*N[1,0])
        c = np.sqrt(beta_v)/(1-beta_tilde_v)*(omega1*N[0,1]+omega2*N[1,1])
        g_out_vec = numerator_vec(omega, V, beta, a, b, c, d)
        constant_piece = np.exp(-1/2 * omega @ gamma @ N.T @ omega)/int_1(a,b,c,d)
        expectation_xi += constant_piece*np.outer(g_out_vec, g_out_vec)

    constant_piece2 = 1/(2*(1-beta_tilde_v))*np.exp(-1/2 * beta_v)*np.sqrt(np.linalg.det(N))*np.exp(1/2*beta_v/(1-beta_tilde_v)**2*M[1,1])
    q_hat = alpha * constant_piece2 * (expectation_xi / samples) 

    return q_hat

def integrand_xi(xi_1, xi_2, constant_piece, sqrtq, gamma, V_inv, M, N, a, d, beta_tilde_v, beta):
    beta_u, beta_v = beta
    xi = np.array([xi_1, xi_2])
    omega = sqrtq @ xi
    omega1, omega2 = omega
    b = np.sqrt(beta_u)*(omega1*N[0,0]+omega2*N[1,0])
    c = np.sqrt(beta_v)/(1-beta_tilde_v)*(omega1*N[0,1]+omega2*N[1,1])
    nu_mu_integral = int_1(a,b,c,d)
    exponential_term = -1/2*omega @ gamma @ M @ V_inv @ omega
    
    return 1/(2*np.pi)*np.exp(-1/2* (xi_1**2 + xi_2**2)) * np.exp(exponential_term)*nu_mu_integral * (np.log(constant_piece) + exponential_term + np.log(nu_mu_integral))

def free_entropy_int(alpha, beta, q, q_hat, mult=10, eps=1e-6):
    beta_u, beta_v = beta
    first_terms = -np.trace(q @ q_hat)/2 -1/2*np.log(np.linalg.det(q_hat + np.eye(2)))+np.trace(q_hat)/2

    sqrt_q = sqrtm(q)
    V = np.eye(2) - q
    gamma = gamma_matrix(beta_v)
    V_inv = np.linalg.inv(V+np.eye(2)*eps)
    M = np.linalg.inv(V_inv+gamma+np.eye(2)*eps)
    N = V_inv @ M
    beta_tilde_v = beta_tilde(beta_v)
    a = beta_u*(1-M[0,0])
    d = np.sqrt(beta_u*beta_v)/(1-beta_tilde_v)*M[0,1]
    constant_piece = 1/2/(1-beta_tilde_v)*np.exp(-1/2 * beta_v)*np.sqrt(np.linalg.det(N))* np.exp(1/2*beta_v/(1-beta_tilde_v)**2*M[1,1])
    xi_integral, error = dblquad(
        integrand_xi,
        -mult, mult,  # xi_1 limits
        lambda xi_1: -mult, lambda xi_1: mult,  # xi_2 limits
        args=(constant_piece, sqrt_q, gamma, V_inv, M, N, a, d, beta_tilde_v, beta),
        epsabs=1e-6,  # Absolute error tolerance
        epsrel=1e-6   # Relative error tolerance
    )
    if error > 1e-6:
        print(f"Warning: High error in integral calculation: {error}")
    if np.isnan(xi_integral):
        print("Warning: Integral calculation returned NaN, check the parameters and integral limits.")
    if np.isinf(xi_integral):
        print("Warning: Integral calculation returned infinity, check the parameters and integral limits.")


    channel_term = -alpha/2*np.log(2)+alpha*constant_piece * xi_integral 


    return first_terms + channel_term


def check_matrix_convergence(Q_old: np.ndarray, 
                             Q_new: np.ndarray, 
                             tolerance: float) -> bool:
    """Check if all elements in Q_new and Q_old are within the given tolerance."""
    diff = np.abs(Q_new - Q_old)
    return np.all(diff < tolerance)

def main(alpha, beta, q_init, samples, iter, damping=.7, tolerance=1e-8):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    q = q_init
    converged = False

    if rank == 0:
        q_list = []
        q_hat_list = []


    for i in range(iter):
        q_hat = q_hat_func_MCMC(alpha, beta, q, samples)
        
        if rank == 0 and i % 1 == 0:
            print(f"[Iter {i}] q =\n{q}")
            print(f"[Iter {i}] q_hat =\n{q_hat}")


        if rank != 0:
            comm.send(q_hat, dest=0)
            q = comm.recv(source=0)
            if q is None:
                break
        
        if rank == 0:
            q_hat_all = np.zeros((size, 2, 2), dtype=np.float64)    

            q_hat_all[0] = q_hat
            for j in range(1, size):
                q_hat_all[j] = comm.recv(source=j)

            q_hat = np.mean(q_hat_all, axis=0)
            old_q = q.copy()
            q = damping*q_func(q_hat) + (1-damping)*q
            q_list.append(q)
            q_hat_list.append(q_hat)
            beta_str = f"{beta[0]}_{beta[1]}"
            q_init_str = f"qinit_{q_init[0,0]:.3f}_{q_init[1,1]:.3f}_{q_init[0,1]:.3f}"
            filename_q = f"data_whitened_uncorrelated/q_list_alpha_{alpha}_samples_{int(size * samples)}_beta_{beta_str}_damping_{damping}_q_init_{q_init_str}_tolerance_{tolerance}.npy"
            np.save(filename_q, q_list)
            filename_q_hat = f"data_whitened_uncorrelated/q_hat_list_alpha_{alpha}_samples_{int(size * samples)}_beta_{beta_str}_damping_{damping}_q_init_{q_init_str}_tolerance_{tolerance}.npy"
            np.save(filename_q_hat, q_hat_list)

            if check_matrix_convergence(old_q, q, tolerance=tolerance):
                print(f"Convergence reached at iteration {i} with q =\n{q}")
                converged = True
                for j in range(1, size):
                    comm.send(None, dest=j)
                break
            else:
                for j in range(1, size):
                    comm.send(q, dest=j)

    if rank == 0 and not converged:
        print("Warning: Maximum iterations reached without convergence.")
        print(f"Alpha = {alpha}")
        print(f"Final q =\n{q}")
        print(f"Final q_hat =\n{q_hat}")

        

if __name__=="__main__":
    alpha = float(sys.argv[1])
    beta_u = float(sys.argv[2])
    beta_v = float(sys.argv[3])
    iter = int(sys.argv[4])
    samples = int(sys.argv[5])

    q_init = np.array([[.1,.0],[.0,.1]])
    beta = (beta_u, beta_v)

    alpha_values = np.linspace(10, 13, 5)
    # alpha = 80
    # alpha_values = [alpha]
    final_q = []
    for alpha in alpha_values:
        print(f"Running for alpha = {alpha}")
        main(alpha, beta, q_init, samples, iter, damping = 0.7)
        

    # final_Q = np.array(final_Q)

    # print(final_Q.shape)

    # plt.plot(alpha_values, final_Q[:, 0], label="$Q_{00}$")
    # plt.plot(alpha_values, final_Q[:, 1], label="$Q_{11}$")
    # plt.plot(alpha_values, final_Q[:, 2], label="$Q_{01}$")
    # plt.xlabel(r"$\alpha$")
    # plt.ylabel("Final Q components")
    # plt.title("Fixed-point Q vs $\\alpha$")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # # ✅ Save the figure to a file
    # plt.savefig("fixed_point_Q_vs_alpha.png", dpi=300)
    