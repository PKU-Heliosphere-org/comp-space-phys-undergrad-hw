import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def dispersion_relation(omega, k, beta):
    """
    Dispersion relation for Alfvén-cyclotron waves
    
    Parameters:
    omega : float
        Wave frequency (normalized to proton cyclotron frequency)
    k : float
        Wave number (normalized to proton inertial length)
    beta : float
        Plasma beta (ratio of thermal to magnetic pressure)
    
    Returns:
    float
        Value of the dispersion relation (should be zero for solutions)
    """
    return omega**2 - k**2 * (1 + beta/2) / (1 + beta/2 + omega**2)

def find_omega(k, beta):
    """
    Find the frequency for a given wave number using the dispersion relation
    """
    omega_initial_guess = k  # Initial guess (Alfvén wave in low beta plasma)
    omega = fsolve(lambda w: dispersion_relation(w, k, beta), omega_initial_guess)[0]
    return omega

def plot_dispersion_relation(beta_values):
    """
    Plot the dispersion relation for different beta values
    """
    k_values = np.linspace(0, 5, 1000)
    
    plt.figure(figsize=(10, 6))
    
    for beta in beta_values:
        omega_values = [find_omega(k, beta) for k in k_values]
        plt.plot(k_values, omega_values, label=f'β = {beta}')
    
    plt.plot(k_values, k_values, 'k--', label='Light line (ω = k)')
    plt.xlabel('k (normalized to proton inertial length)')
    plt.ylabel('ω (normalized to proton cyclotron frequency)')
    plt.title('Dispersion Relation for Alfvén-Cyclotron Waves')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.savefig('dispersion_relation.png')
    plt.show()

if __name__ == "__main__":
    beta_values = [0.01, 0.1, 1.0]
    plot_dispersion_relation(beta_values)
    print("Dispersion relation plot has been saved as 'dispersion_relation.png'")