import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
try:
    from solar_wind_cpp_core import SolarWindCore
    USE_CPP = True
except ImportError:
    print("C++ module not found. Using pure Python implementation.")
    USE_CPP = False

class SolarWindIonSimulation:
    def __init__(self, params=None):
        # Default parameters
        self.params = {
            'N': 51,  # Number of wave modes
            'epsilon_0': 0.1,  # Wave energy density
            'beta': 0.1,  # Plasma beta
            'omega_range': [0.01, 0.4],  # Wave frequency range (normalized to Ωp)
            'dt': 0.01,  # Time step (normalized to Ωp^-1)
            'n_particles': 10000,  # Number of test particles
            'tau': 200,  # Time scale for wave amplitude variation
            'delta_tau': 50,  # Width of wave amplitude variation
            'z_range': [0, 3000],  # Spatial range for initial particle distribution
            'simulation_time': 20000  # Total simulation time
        }
        if params:
            self.params.update(params)
        
        # Initialize wave parameters
        self.initialize_waves()
        
        # Initialize particles
        self.initialize_particles()
        
        if USE_CPP:
            # Initialize C++ core
            self.cpp_core = SolarWindCore(
                self.omega,
                self.k_z,
                self.B_j,
                self.params['epsilon_0'],
                self.params['tau'],
                self.params['delta_tau']
            )

    def initialize_waves(self):
        """Initialize wave parameters including frequencies and wave numbers"""
        # Calculate frequencies for each wave mode
        log_omega = np.logspace(
            np.log10(self.params['omega_range'][0]),
            np.log10(self.params['omega_range'][1]),
            self.params['N']
        )
        self.omega = log_omega
        
        # Calculate wave numbers using dispersion relation
        self.k_z = np.zeros_like(self.omega)
        for i, omega in enumerate(self.omega):
            # Simplified dispersion relation for Alfvén-cyclotron waves
            self.k_z[i] = omega * (1 + 1/(1 - omega**2))**0.5
        
        # Initialize wave amplitudes with power-law spectrum
        alpha = 5/3  # Spectral index
        self.B_j = np.zeros(self.params['N'])
        
        # Calculate unnormalized amplitudes
        B_j_unnormalized = self.omega**(-alpha/2)
        
        # Normalize to maintain total energy
        total_energy = self.params['epsilon_0']
        normalization_factor = np.sqrt(2 * total_energy / np.sum(B_j_unnormalized**2))
        self.B_j = normalization_factor * B_j_unnormalized

    def wave_amplitude_time_dependence(self, t):
        """Calculate time-dependent wave amplitude"""
        tau = self.params['tau']
        delta_tau = self.params['delta_tau']
        
        if t < tau:
            return self.params['epsilon_0'] * np.exp(-(t - tau)**2 / delta_tau**2)
        return self.params['epsilon_0']
    
    def initialize_particles(self):
        """Initialize test particles with Maxwell distribution"""
        n_particles = self.params['n_particles']
        
        # Initialize positions uniformly in z-range
        self.z = np.random.uniform(
            self.params['z_range'][0],
            self.params['z_range'][1],
            n_particles
        )
        
        # Initialize velocities with Maxwell distribution
        v_thermal = np.sqrt(self.params['beta'])  # Thermal velocity (normalized to VA)
        self.vx = np.random.normal(0, v_thermal, n_particles)
        self.vy = np.random.normal(0, v_thermal, n_particles)
        self.vz = np.random.normal(0, v_thermal, n_particles)

    def calculate_fields(self, z, t):
        """Calculate electromagnetic fields at given position and time"""
        if USE_CPP:
            return self.cpp_core.calculate_fields(z, t)
        else:
            # Initialize fields
            Bx, By, Ex, Ey = 0, 0, 0, 0
            
            # Sum over all wave modes
            for j in range(self.params['N']):
                phase = self.k_z[j] * z - self.omega[j] * t
                amplitude = self.B_j[j] * self.wave_amplitude_time_dependence(t)
                
                # Left-hand circular polarization in x-y plane
                Bx += amplitude * np.cos(phase)
                By += amplitude * np.sin(phase)
                
                # Electric field components with proper dispersion relation
                # For Alfvén-cyclotron waves, E/B = omega/k
                E_amplitude = amplitude * (self.omega[j] / self.k_z[j])
                Ex += -E_amplitude * np.sin(phase)
                Ey += E_amplitude * np.cos(phase)
            
            return Ex, Ey, Bx, By

    def boris_push(self, z, v, t, dt):
        """Implement Boris algorithm for particle push"""
        if USE_CPP:
            return self.cpp_core.boris_push(z, v.tolist(), t, dt)
        else:
            # Constants for O5+ ion (normalized to proton values)
            q_m_ratio = 5/16  # Charge-to-mass ratio for O5+
            
            # Get fields at current position
            Ex, Ey, Bx, By = self.calculate_fields(z, t)
            Bz = 1.0  # Background field
            Ez = 0.0  # No parallel electric field
            
            # Construct field vectors
            E = np.array([Ex, Ey, Ez])
            B = np.array([Bx, By, Bz])
            
            # Boris algorithm implementation
            # 1. Half acceleration in E field
            v_minus = v + 0.5 * dt * q_m_ratio * E
            
            # 2. Magnetic field rotation
            t_vec = 0.5 * dt * q_m_ratio * B
            t_squared = np.sum(t_vec**2)
            
            # Boris rotation factors
            s_vec = 2 * t_vec / (1 + t_squared)
            
            v_prime = v_minus + np.cross(v_minus, t_vec)
            v_plus = v_minus + np.cross(v_prime, s_vec)
            
            # 3. Half acceleration in E field
            new_v = v_plus + 0.5 * dt * q_m_ratio * E
            
            # 4. Position update (using average of old and new velocities)
            new_z = z + 0.5 * dt * (v[2] + new_v[2])
            
            return new_z, new_v

    def run_simulation_multi_particle(self):
        """Run simulation for multiple particles using Boris algorithm"""
        n_particles = self.params['n_particles']
        dt = self.params['dt']
        t = np.arange(0, self.params['simulation_time'], dt)
        n_steps = len(t)
        
        # Initialize array to store results for all particles
        all_solutions = np.zeros((n_particles, n_steps, 4))
        
        # record the start time
        start_time = time.time()

        # Run simulation for each particle
        for i in range(n_particles):
            if i % 1 == 0:
                print(f"Running simulation for particle {i+1}/{n_particles}")
                # calculate and print the elapsed time for completing i particles movers
                elapsed_time = time.time() - start_time
                print(f"Time elapsed for particle {i+1}: {elapsed_time:.2f} seconds")

            # Initial conditions for current particle
            z = self.z[i]
            v = np.array([self.vx[i], self.vy[i], self.vz[i]])
            
            # Store initial state
            all_solutions[i, 0] = [z, v[0], v[1], v[2]]
            
            # Time integration using Boris algorithm
            for j in range(1, n_steps):
                if j % 500000 == 0:
                    print(f"Particle {i+1}: Time step {j}/{n_steps}")
                z, v = self.boris_push(z, v, t[j-1], dt)
                if isinstance(v, list):  # Convert C++ list to numpy array if needed
                    v = np.array(v)
                all_solutions[i, j] = [z, v[0], v[1], v[2]]
        
        return t, all_solutions

    def analyze_results(self, t, solution):
        """Analyze and plot simulation results"""
        # Create figure for 3D trajectory
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(solution[:, 1], solution[:, 2], solution[:, 0])
        ax1.set_xlabel('vx (VA)')
        ax1.set_ylabel('vy (VA)')
        ax1.set_zlabel('z (VA/Ωp)')
        ax1.set_title('Particle Trajectory')
        
        # Plot velocity components
        ax2 = fig.add_subplot(132)
        ax2.plot(t, solution[:, 1], label='vx')
        ax2.plot(t, solution[:, 2], label='vy')
        ax2.plot(t, solution[:, 3], label='vz')
        ax2.set_xlabel('Time (Ωp^-1)')
        ax2.set_ylabel('Velocity (VA)')
        ax2.set_title('Velocity Components')
        ax2.legend()
        
        # Plot magnetic moment
        v_perp_squared = solution[:, 1]**2 + solution[:, 2]**2
        magnetic_moment = 0.5 * v_perp_squared  # Normalized magnetic moment
        ax3 = fig.add_subplot(133)
        ax3.plot(t, magnetic_moment)
        ax3.set_xlabel('Time (Ωp^-1)')
        ax3.set_ylabel('Magnetic Moment (normalized)')
        ax3.set_title('Magnetic Moment')
        
        plt.tight_layout()
        plt.show()

    def plot_velocity_distribution(self, all_solutions, times):
        """Plot velocity distribution at different times"""
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Ion Velocity Distribution at Different Times')
        
        for i, time in enumerate(times):
            time_index = int(time / self.params['dt'])
            vx = all_solutions[:, time_index, 1]
            vy = all_solutions[:, time_index, 2]
            vz = all_solutions[:, time_index, 3]
            v_perp = np.sqrt(vx**2 + vy**2)
            
            # vx-vz plane
            axs[0, i].hist2d(vz, vx, bins=50, cmap='jet')
            axs[0, i].set_xlabel('vz (VA)')
            axs[0, i].set_ylabel('vx (VA)')
            axs[0, i].set_title(f'Time = {time} Ωp^-1')
            
            # vz-v_perp plane
            axs[1, i].hist2d(vz, v_perp, bins=50, cmap='jet')
            axs[1, i].set_xlabel('vz (VA)')
            axs[1, i].set_ylabel('v_perp (VA)')
        
        plt.tight_layout()
        plt.show()

    def plot_energy_conservation(self, t, solution, num_particles=5):
        """Plot energy conservation for selected particles"""
        plt.figure(figsize=(10, 6))
        
        # Select random particles to analyze
        indices = np.random.choice(solution.shape[0], num_particles, replace=False)
        
        for idx in indices:
            particle_solution = solution[idx]
            energy = self.analyze_energy_conservation(particle_solution)
            # Normalize to initial energy for better comparison
            normalized_energy = energy / energy[0]
            plt.plot(t, normalized_energy, label=f'Particle {idx}')
        
        plt.xlabel('Time (Ωp^-1)')
        plt.ylabel('Normalized Energy')
        plt.title('Energy Conservation with Boris Algorithm')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Set parameters for Case 1
    params = {
        'N': 51,
        'epsilon_0': 0.1,
        'beta': 0.1,
        'omega_range': [0.01, 0.4],
        'simulation_time': 2000,  # Reduced from 20000
        'n_particles': 100  # Reduced from 10000
    }
    
    # Create simulation instance
    sim = SolarWindIonSimulation(params)
    
    # Run simulation for multiple particles
    t, all_solutions = sim.run_simulation_multi_particle()
    
    # Plot velocity distribution at different times
    times_to_plot = [2500, 10000, 15000, 20000]
    sim.plot_velocity_distribution(all_solutions, times_to_plot)
    
    # Analyze energy conservation for a few particles
    sim.plot_energy_conservation(t, all_solutions, num_particles=5)