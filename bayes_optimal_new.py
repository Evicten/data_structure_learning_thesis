import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
from typing import Callable, Tuple
import sys
from scipy.integrate import quad
import plotly.graph_objects as go
import plotly.io as pio
import os


def integrand_xi(xi, delta, q):
    return 1/np.sqrt(2*np.pi)*np.exp(-xi**2/2) * np.sinh(np.sqrt(q/delta)*xi)*np.tanh(np.sqrt(q/delta)*xi)


def calculate_q_hat(alpha: float, q: float, delta: float) -> float:

    bad_erros = []
    xi_integral, error = quad(integrand_xi, -20, 20, args=(delta, q))

    if error > 1e-6:
        bad_erros.append(error)
    

    if len(bad_erros) > 0:
        print("Bad errors: ", len(bad_erros))
    
    return alpha /(2*delta) * np.exp(-q/(2*delta)) * xi_integral

def update_q(q_hat: float) -> float:
    """Calculate new q"""
    
    q = q_hat/(1+ q_hat)

    return q

def check_convergence(old_val: float, 
                     new_val: float, 
                     tolerance: float) -> bool:
    """Check if the iteration has converged within specified tolerance"""
    return abs(new_val - old_val) < tolerance

def fixed_point_iteration(initial_q: float, alpha: float, delta: float, tolerance: float = 1e-8, max_iterations: int = 2000, damping: float = 0.2, verbose=False) -> Tuple[float, bool]:
    """Main iteration loop to find fixed point"""
    
    q = initial_q
    
    for iteration in range(max_iterations):
        # Store old values for convergence check
        old_vals = q
        
        # Calculate hat values using your simplified versions of (7), (8), (9)
        q_hat = calculate_q_hat(alpha=alpha, q=q, delta=delta)  # Example values
        
        # Update m, q, gamma using equations (4), (5), (6)
        q = update_q(q_hat)  # Example values

        q = (1-damping)*q + damping*old_vals
        
        # Check for convergence
        if check_convergence(old_vals, q, tolerance):
            
            print(f"Converged after {iteration + 1} iterations")
            return q, True
        
        # Optional: Print current values
        if iteration % 10 == 0 and verbose:
            print(f"Iteration {iteration}: q={q:.6f}")
    
    print("Warning: Maximum iterations reached without convergence")
    print(f"Final q: {q:.6f}, alpha: {alpha}, delta: {delta}")
    return q, False

def pca_result(alpha: float, delta: float) -> float:
    """Calculate PCA result based on alpha and delta"""
    if alpha <= 4 * delta**2:
        return 0.0
    if alpha > 4 * delta**2:
        return np.sqrt((alpha - 4 * delta**2) / (alpha + 2 * delta))

if __name__ == "__main__":
    # Get parameters from command line arguments or use defaults
    if len(sys.argv) != 3:
        print("Usage: python script.py initial_m initial_q initial_gamma lambda delta rho")
        sys.exit(1)
    
    # Parse command line arguments
    initial_q = float(sys.argv[1])
    delta = float(sys.argv[2])
    
    alpha_values = np.linspace(0.01, 30, 2000)
    delta_values = [0.5]

    for delta in delta_values:
        final_q_list = []
        converged_alphas = []

        # Compute final_q for each alpha
        for alpha in alpha_values:
            final_q, converged = fixed_point_iteration(initial_q, alpha, delta, damping=0.3)
            final_q_list.append(final_q)
            converged_alphas.append(converged)

        filtered_alpha_values = np.array(alpha_values)[np.array(converged_alphas)]
        filtered_q_values = np.array(final_q_list)[np.array(converged_alphas)]


        filename = f"data/delta={delta:.3f}_initial_q={initial_q:.2f}.npz"
        if os.path.exists(filename):
            # Load existing data
            existing = np.load(filename)
            old_alpha = existing['alpha']
            old_q = existing['q']

            # Concatenate
            all_alpha = np.concatenate([old_alpha, filtered_alpha_values])
            all_q = np.concatenate([old_q, filtered_q_values])

        else:
            all_alpha = filtered_alpha_values
            all_q = filtered_q_values

        # Save (overwrites file, but now with appended content)
        np.savez(filename, alpha=all_alpha, q=all_q)

#  # Create the figure
# fig = go.Figure()

# colors = ['blue', 'green', 'purple']

# for idx, delta in enumerate(delta_values):
#     final_q_list = []
#     pca_m_list = []
    
#     # Compute final_m for each alpha
#     for alpha in alpha_values:
#         final_q = fixed_point_iteration(initial_q, alpha, delta, damping=0.1)
#         final_q_list.append(final_q)
#         pca_m = pca_result(alpha, delta)
#         pca_m_list.append(pca_m)

#     # PCA m
#     #m_pca = np.sqrt((alpha_values-4*delta**2)/(alpha_values+2*delta))

#     # Compute alpha_theory
#     alpha_theory = 2 * delta**2

   

#     # Add the main line plot
#     fig.add_trace(go.Scatter(x=alpha_values, y=np.sqrt(final_q_list), mode='lines', name='sqrt(q) vs Alpha BO delta = ' + str(delta)))
#     fig.add_trace(go.Scatter(x=alpha_values, y=pca_m_list, mode='lines', name='PCA m vs Alpha delta = ' + str(delta), line=dict(dash='dash')))

#     # Add the PCA line
#     #fig.add_trace(go.Scatter(x=alpha_values, y=m_pca, mode='lines', name='m vs Alpha PCA'))

#     # Add the red dot at (alpha_theory, 0)
#     fig.add_trace(go.Scatter(x=[alpha_theory], y=[0], mode='markers',
#                              marker=dict(color=colors[idx], size=10), 
#                              name='Linear Stability Alpha, delta = ' + str(delta)))

# # Customize layout
# fig.update_layout(
#     title=f"cosine similarity vs. Alpha Bayes-Optimal and PCA",
#     xaxis_title="Alpha",
#     yaxis_title="q",
#     template="plotly_white"
# )

# # Save the figure
# plot_name = f"figures/q_vs_alpha_BO.png"
# fig.write_image(plot_name)  # Save as PNG
# print("Plot saved as " + plot_name)

# # Show the figure (optional)
# fig.show()

   