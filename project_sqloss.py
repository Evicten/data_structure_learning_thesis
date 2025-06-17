import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
from typing import Callable, Tuple
import sys
import os

def expectation_ys(func: Callable[[int, int], float]) -> float:
    """Calculate the expectation of a function with respect to (y,s)"""
    return 1/4*func(1,1) + 1/4*func(1,-1) + 1/4*func(0,1) + 1/4*func(0,-1)

def expectation_constant(inf: float, sup: float, m: float, q: float, b: float, delta: float) -> Callable[[int, int], float]:
    """Expectation of 1 over r.v. h ~ N(ysm+b, delta*q) between a and b. Returns a function that depends only on y and s"""
    if inf == -np.inf and sup == np.inf:
        def func(y: int, s: int) -> float:
            return 1
    elif inf == np.inf and sup == np.inf:
        def func(y: int, s: int) -> float:
            return 0
    elif inf == -np.inf:
        def func(y: int, s: int) -> float:
            return (1-1/2*erfc((sup*y-y*s*m-b)/np.sqrt(2*q*delta)))
    elif sup == np.inf:
        def func(y: int, s: int) -> float:
            return 1/2*(erfc((inf*y-y*s*m-b)/np.sqrt(2*q*delta)))
    else:
        def func(y: int, s: int) -> float:
            return 1/2*(erfc((inf*y-y*s*m-b)/np.sqrt(2*q*delta)) - erfc((sup*y-y*s*m-b)/np.sqrt(2*q*delta)))
    return func

def expectation_h(inf: float, sup: float, m: float, q: float, b: float, delta: float) -> Callable[[int, int], float]:
    """Expectation of h over r.v. h ~ N(ysm+b, delta*q) between a and b. Returns a function that depends only on y and s"""
    if inf == -np.inf and sup == np.inf:
        def func(y: int, s: int) -> float:
            return y*s*m+b
    elif inf == np.inf and sup == np.inf:
        def func(y: int, s: int) -> float:
            return 0
    elif inf == -np.inf:
        def func(y: int, s: int) -> float:
            return (y*s*m+b)*(1-1/2*erfc((sup*y-y*s*m-b)/np.sqrt(2*q*delta))) - np.sqrt(q*delta/(2*np.pi))*np.exp(-(sup*y-y*s*m-b)**2/(2*q*delta))
    elif sup == np.inf:
        def func(y: int, s: int) -> float:
            return 1/2*(y*s*m+b)*erfc((inf*y-y*s*m-b)/np.sqrt(2*q*delta)) + np.sqrt(q*delta/(2*np.pi))*np.exp(-(inf*y-y*s*m-b)**2/(2*q*delta))
    else:
        def func(y: int, s: int) -> float:
            return 1/2*(y*s*m+b)*(erfc((inf*y-y*s*m-b)/np.sqrt(2*q*delta)) - erfc((sup*y-y*s*m-b)/np.sqrt(2*q*delta))) + np.sqrt(q*delta/(2*np.pi))*(np.exp(-(inf*y-y*s*m-b)**2/(2*q*delta)) - np.exp(-(sup*y-y*s*m-b)**2/(2*q*delta)))
    return func

def expectation_h2(inf: float, sup: float, m: float, q: float, b: float, delta: float) -> Callable[[int, int], float]:
    """Expectation of h^2 over r.v. h ~ N(ysm+b, delta*q) between a and b. Returns a function that depends only on y and s"""
    if inf == -np.inf and sup == np.inf:
        def func(y: int, s: int) -> float:
            return q*delta + (y*s*m+b)**2 
    elif inf == np.inf and sup == np.inf:
        def func(y: int, s: int) -> float:
            return 0
    elif inf == -np.inf:
        def func(y: int, s: int) -> float:
            return -np.sqrt(q*delta)/np.sqrt(2*np.pi)*(sup*y-y*s*m-b)*np.exp(-(sup*y-y*s*m-b)**2/(2*q*delta))+(q*delta-(y*s*m+b)**2)*expectation_constant(inf, sup, m, q, b, delta)(y, s)+2*(y*s*m+b)*expectation_h(inf, sup, m, q, b, delta)(y, s)
    elif sup == np.inf:
        def func(y: int, s: int) -> float:
            return np.sqrt(q*delta)/np.sqrt(2*np.pi)*(inf*y-y*s*m-b)*np.exp(-(inf*y-y*s*m-b)**2/(2*q*delta))+(q*delta-(y*s*m+b)**2)*expectation_constant(inf, sup, m, q, b, delta)(y, s)+2*(y*s*m+b)*expectation_h(inf, sup, m, q, b, delta)(y, s)
    else:
        def func(y: int, s: int) -> float:
            return np.sqrt(q*delta/(2*np.pi))*((inf*y-y*s*m-b)*np.exp(-(inf*y-y*s*m-b)**2/(2*q*delta)) - (sup*y-y*s*m-b)*np.exp(-(sup*y-y*s*m-b)**2/(2*q*delta)))+2*(y*s*m+b)*expectation_h(inf, sup, m, q, b, delta)(y, s)+(q*delta-(y*s*m+b)**2)*expectation_constant(inf, sup, m, q, b, delta)(y, s)
        
    return func

def global_minimizer(V: float) -> float:
    return 1-np.sqrt(1+2*V)
        
        
def calculate_m_hat(m: float, q: float, V: float, b: float, alpha: float, delta: float) -> float:
    global_min = global_minimizer(V)
    constant_piece = 2*alpha/(1+2*V)
    expectation = expectation_ys(lambda y, s: y*s*(y*expectation_constant(global_min, np.inf, m, q, b, delta)(y,s)-expectation_h(global_min, np.inf, m, q, b, delta)(y,s)))
    return constant_piece*expectation


def calculate_q_hat(m: float, q: float, V: float, b: float, alpha: float, delta: float) -> float:
    global_min = global_minimizer(V)
    constant_piece = 4*alpha*delta/(1+2*V)**2
    expectation = expectation_ys(lambda y, s: (y*expectation_constant(global_min, np.inf, m, q, b, delta)(y,s)-2*y*expectation_h(global_min, np.inf, m, q, b, delta)(y,s)+expectation_h2(global_min, np.inf, m, q, b, delta)(y,s)))
    return constant_piece*expectation

def calculate_V_hat(m: float, q: float, V: float, b: float, alpha: float, delta: float) -> float:
    global_min = global_minimizer(V)
    constant_piece = 2*alpha*delta/(1+2*V)
    expectation = expectation_ys(lambda y, s: expectation_constant(global_min, np.inf, m, q, b, delta)(y,s))
    return constant_piece*expectation

def calculate_b(m: float, q: float, V: float, b: float, delta: float) -> float:
    global_min = global_minimizer(V)
    return expectation_ys(lambda y, s: expectation_h(-np.inf, global_min, m, q, b, delta)(y,s)+1/(1+2*V)*(expectation_h(global_min, np.inf, m, q, b, delta)(y,s)+expectation_constant(global_min, np.inf, m, q, b, delta)(y,s)))

def update_m_q_V(m_hat: float, q_hat: float, V_hat: float, lambda_val: float, delta: float, regularisation = True) -> Tuple[float, float, float]:
    """Calculate new m, q, gamma using equations (4), (5), (6)"""
    if regularisation:
        denominator = lambda_val + V_hat + 1e-6
    else:
        denominator = V_hat + lambda_val
    
    
    m = m_hat/denominator
    q = (q_hat+m_hat**2)/denominator**2
    V = delta/denominator
    
    return m, q, V

def generalisation_error_calculation(m:float, q: float, b: float, delta: float) -> float:
    return 1/8*erfc((m+b)/np.sqrt(2*q*delta)) + 1/8*erfc((b-m)/np.sqrt(2*q*delta))+1/4*erfc(-b/np.sqrt(2*q*delta))

def generalisation_error_calculation_cutoff(m:float, q: float, b: float, delta: float, cutoff: float) -> float:
    return 1/8*erfc((m+b-cutoff)/np.sqrt(2*q*delta)) + 1/8*erfc((b-cutoff-m)/np.sqrt(2*q*delta))+1/4*erfc((-b+cutoff)/np.sqrt(2*q*delta))

def check_convergence(old_vals: Tuple[float, float, float], 
                     new_vals: Tuple[float, float, float], 
                     tolerance: float, verbose: bool = False) -> bool:
    """Check if the iteration has converged within specified tolerance"""
    return all(abs(new - old) < tolerance for new, old in zip(new_vals, old_vals))

def fixed_point_iteration(initial_m: float, initial_q: float, initial_V: float,
                         lambda_val: float, alpha: float, delta: float, initial_b: float,
                         tolerance: float = 1e-8, max_iterations: int = 2000, damping: float = 0.5, verbose: bool = False, regularisation: bool = True) -> Tuple[float, float, float, float]:
    """Main iteration loop to find fixed point"""
    
    m, q, V = initial_m, initial_q, initial_V
    b = initial_b
    m_hat, q_hat, V_hat = 0, 0, 0
    
    for iteration in range(max_iterations):
        # Store old values for convergence check
        old_vals = (m, q, V, b)
        
        m_hat = calculate_m_hat(m, q, V, b, alpha, delta)
        q_hat = calculate_q_hat(m, q, V, b, alpha, delta)
        V_hat = calculate_V_hat(m, q, V, b, alpha, delta)
        # Update m, q, V using equations (4), (5), (6)
        m, q, V= update_m_q_V(m_hat, q_hat, V_hat, lambda_val, delta, regularisation=regularisation)
        #b = calculate_b(m, q, V, b, delta)

        m, q, V, b = (1-damping)*m + damping*old_vals[0], (1-damping)*q + damping*old_vals[1], (1-damping)*V + damping*old_vals[2], (1-damping)*b + damping*old_vals[3]
        
        # Check for convergence
        if check_convergence(old_vals, (m, q, V), tolerance, verbose=verbose):
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            return m, q, V, V_hat, b, True
        
        # Optional: Print current values
        # if verbose==True and iteration % 100 == 0:
        #     print(f"Iteration {iteration}: m={m:.9f}, q={q:.9f}, V={V:.9f}")
    
    print("Warning: Maximum iterations reached without convergence")
    print(f"m={m:.6f}, q={q:.6f}, V={V:.6f}, b={b:.6f}, alpha={alpha:.6f}, delta={delta:.6f}")
    return m, q, V, V_hat, b, False

if __name__ == "__main__":
    # Get parameters from command line arguments or use defaults
    if len(sys.argv) != 7:
        print("Usage: python script.py initial_m initial_q initial_gamma lambda delta rho")
        sys.exit(1)
    
    # Parse command line arguments
    initial_m = float(sys.argv[1])
    initial_q = float(sys.argv[2])
    initial_V = float(sys.argv[3])
    lambda_val = float(sys.argv[4])
    delta = float(sys.argv[5])
    initial_b = float(sys.argv[6])
    
    alpha_values = np.linspace(0.01, 25, 1000)
    #alpha_values = [10]
    #delta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    delta_values = [delta]

    for delta in delta_values:
        list_of_mqVb = []
        converged_alpha = []
        for alpha in alpha_values:
            final_m, final_q, final_V, final_V_hat, final_b, converged = fixed_point_iteration(
            initial_m, initial_q, initial_V,
            lambda_val, alpha, delta, initial_b, damping=0.9, max_iterations=30000, verbose=True, regularisation=True, tolerance=1e-8)
            list_of_mqVb.append((final_m, final_q, final_V, final_V_hat, final_b))
            converged_alpha.append(converged)

        filtered_alpha_values = [alpha for alpha, converged in zip(alpha_values, converged_alpha) if converged]
        filtered_m_values = [mqvb[0] for mqvb, converged in zip(list_of_mqVb, converged_alpha) if converged]
        filtered_q_values = [mqvb[1] for mqvb, converged in zip(list_of_mqVb, converged_alpha) if converged]
        filtered_V_values = [mqvb[2] for mqvb, converged in zip(list_of_mqVb, converged_alpha) if converged]
        filtered_V_hat_values = [mqvb[3] for mqvb, converged in zip(list_of_mqVb, converged_alpha) if converged]
        
        filename = f"data/delta={delta:.3f}_lambda={lambda_val}_initial_m={initial_m:.2f}_initial_q={initial_q:.2f}_initial_V={initial_V:.2f}.npz"
        
        if os.path.exists(filename):
            # Load existing data
            existing = np.load(filename)
            old_alpha = existing['alpha']
            old_m = existing['m']
            old_q = existing['q']
            old_V = existing['V']
            old_V_hat = existing['V_hat']

            # Concatenate
            all_alpha = np.concatenate([old_alpha, filtered_alpha_values])
            all_m = np.concatenate([old_m, filtered_m_values])
            all_q = np.concatenate([old_q, filtered_q_values])
            all_V = np.concatenate([old_V, filtered_V_values])
            all_V_hat = np.concatenate([old_V_hat, filtered_V_hat_values])

        else:
            all_alpha = filtered_alpha_values
            all_m = filtered_m_values
            all_q = filtered_q_values
            all_V = filtered_V_values
            all_V_hat = filtered_V_hat_values

        # Save (overwrites file, but now with appended content)
        np.savez(filename, alpha=all_alpha, m=all_m, q=all_q, V=all_V, V_hat=all_V_hat)
    
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_alpha_values, filtered_m_values/np.sqrt(filtered_q_values), color='b')
    plt.xlabel("Alpha")
    plt.ylabel("m")
    plt.title("m vs. Alpha delta = " + str(delta) + ", lambda = " + str(lambda_val))
    plt.grid()
    plot_name = f"figures/m_alpha_delta={delta}_lambda={lambda_val}.png"
    plt.savefig(plot_name)
    print("Plot saved as " + plot_name)

    # for lambda_val in lambda_values:
    #     gen_error_list = []
    #     for alpha in alpha_values:
    #         final_m, final_q, final_V, final_b = fixed_point_iteration(
    #         initial_m, initial_q, initial_V,
    #         lambda_val, alpha, delta, initial_b)
    #         gen_error_list.append(generalisation_error_calculation(final_m, final_q, final_b))
    #     gen_error_lambda.append(gen_error_list)

    
    # Run the fixed point iteration

    # plt.figure(figsize=(10, 6))
    # for i, lambda_val in enumerate(lambda_values):
    #     plt.plot(alpha_values, gen_error_lambda[i], label=f"lambda = {lambda_val}")
    # plt.xlabel("Alpha")
    # plt.ylabel("Generalisation Error")
    # plt.title("Generalisation Error vs. Alpha for different lambda values")
    # plt.grid()
    # plt.legend()
    # plot_name = "figures/project_sqloss_relu" + str(lambda_val) + ".png"
    # plt.savefig(plot_name)
    # print("Plot saved as " + plot_name)