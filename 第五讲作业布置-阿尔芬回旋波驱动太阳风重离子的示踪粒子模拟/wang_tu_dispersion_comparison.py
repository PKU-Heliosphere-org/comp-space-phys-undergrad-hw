import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def wang_cb_2014(k, va2c):
    """
    Wang et al. 2014 dispersion relation
    
    Parameters:
    k : array-like
        Wave number (normalized to Ωp/VA)
    va2c : float
        Square of Alfvén speed normalized to speed of light (VA^2/c^2)
    
    Returns:
    array-like
        Frequency (normalized to Ωp)
    """
    omega = np.zeros_like(k)
    for i, k_val in enumerate(k):
        # Use binary search to find omega
        omega_min, omega_max = 0.001, 0.999
        while omega_max - omega_min > 1e-6:
            omega_mid = (omega_min + omega_max) / 2
            k_test = omega_mid * (va2c + 1/(1 - omega_mid**2))**0.5
            if k_test < k_val:
                omega_min = omega_mid
            else:
                omega_max = omega_mid
        omega[i] = (omega_min + omega_max) / 2
    return omega

def tu_marsch_2001(k):
    """
    Tu and Marsch 2001 dispersion relation
    
    Parameters:
    k : array-like
        Wave number (normalized to Ωp/VA)
    
    Returns:
    array-like
        Frequency (normalized to Ωp)
    """
    omega = np.zeros_like(k)
    for i, k_val in enumerate(k):
        # Use binary search to find omega that satisfies k = (omega^2 / (1-omega))^(1/2)
        omega_min, omega_max = 0.001, 0.999
        while omega_max - omega_min > 1e-6:
            omega_mid = (omega_min + omega_max) / 2
            k_test = (omega_mid**2 / (1 - omega_mid))**0.5
            if k_test < k_val:
                omega_min = omega_mid
            else:
                omega_max = omega_mid
        omega[i] = (omega_min + omega_max) / 2
    return omega

def plot_dispersion_comparison(va2c=0.002, save_dir=None):
    """
    Plot and compare the dispersion relations of Wang et al. 2014 and Tu & Marsch 2001
    
    Parameters:
    va2c : float, optional
        Square of Alfvén speed normalized to speed of light (VA^2/c^2)
    """
    k = np.logspace(-2, 1, 1000)  # Wave number range
    
    omega_wang = wang_cb_2014(k, va2c)
    omega_tu = tu_marsch_2001(k)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(k, omega_wang, 'b-', label='Wang et al. 2014')
    plt.loglog(k, omega_tu, 'r--', label='Tu & Marsch 2001')
    plt.loglog(k, k, 'k:', label='ω = k')
    
    plt.xlabel('k (normalized to Ωp/VA)')
    plt.ylabel('ω (normalized to Ωp)')
    plt.title('Dispersion Relation Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.xlim(1e-2, 1e1)
    plt.ylim(1e-2, 1e0)
    
    if save_dir is None:
        # Use the current working directory (where the script is run from)
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, 'figures', 'dispersion_relation')
        print(f"Figures will be saved to the default directory: {save_dir}")
    
    # Create figures directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'wang_tu_dispersion_comparison_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    
    # Save the figure
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Dispersion relation comparison plot has been saved as '{filepath}'")

if __name__ == "__main__":
    # Plot and save the dispersion relations
    # The default save_dir (None) will use the current working directory
    plot_dispersion_comparison()