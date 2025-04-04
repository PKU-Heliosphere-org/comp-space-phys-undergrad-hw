import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import sys
import h5py
import datetime
from solar_wind_cpp_core_v2 import SolarWindCoreV2

# 创建一个同时输出到屏幕和文件的函数
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open('run.log', 'a') as f:
        print(*args, file=f, **kwargs)

# 确保figures目录存在
os.makedirs('./figures', exist_ok=True)

class SolarWindIonSimulationV2:
    def __init__(self, params=None):
        # Default parameters
        self.params = {
            'N': 51,  # Number of wave modes
            'epsilon_0': 0.4, #0.1,  # Wave energy density
            'beta': 0.1,  # Plasma beta
            'va2c': 0.002, # 0.002,  # Alfvén speed normalized to c (V_A / c)
            'omega_range': [0.01, 0.4],  # Wave frequency range (normalized to Ωp)
            'dt': 0.05,  # Time step (normalized to Ωp^-1)
            'n_particles': 4000, #10000,  # Number of test particles
            'tau': 100, #200,  # Time scale for wave amplitude variation
            'delta_tau': 50,  # Width of wave amplitude variation
            'z_range': [0, 3000],  # Spatial range for initial particle distribution
            'simulation_time': 4000,  # Total simulation time
            'output_sampling': 40,  # Save every Nth timestep
            'batch_size': 500  # Number of particles to process in each batch
        }
        if params:
            self.params.update(params)
        
        # Initialize wave parameters
        self.dispersion_formula_choice = 'TuMarsch2001' #'WangCB2014' or 'TuMarsch2001' or 'Cranmer2014'
        self.initialize_waves()
        
        # Initialize C++ core
        self.cpp_core = SolarWindCoreV2(
            self.omega,
            self.k_z,
            self.B_j,
            self.params['epsilon_0'],
            self.params['tau'],
            self.params['delta_tau'],
            self.params['dt'],
            self.params['beta']
        )

    def plot_dispersion_check(self, show=True, save=True):
        """Plot dispersion relation check for initialized waves"""
        plt.figure(figsize=(10, 8))
        
        # Plot discrete wave modes
        plt.scatter(self.k_z, self.omega, c='blue', label='Wave modes', alpha=0.6)
        
        # Plot continuous dispersion relation for comparison
        k_continuous = np.linspace(0, max(self.k_z), 1000)
        omega_continuous = np.zeros_like(k_continuous)
        
        # Find omega for each k using the dispersion relation
        for i, k in enumerate(k_continuous):
            # Use binary search to find omega that satisfies k = omega * sqrt(1 + 1/(1-omega^2))
            omega_min, omega_max = 0.001, 0.999
            while omega_max - omega_min > 1e-6:
                omega = (omega_min + omega_max) / 2
                k_test = omega * (1 + 1/(1 - omega**2))**0.5
                if k_test < k:
                    omega_min = omega
                else:
                    omega_max = omega
            omega_continuous[i] = (omega_min + omega_max) / 2
        
        plt.plot(k_continuous, omega_continuous, 'r-', label='Theoretical curve', alpha=0.7)
        
        # Plot omega = k line for reference
        k_ref = np.linspace(0, max(self.k_z), 100)
        plt.plot(k_ref, k_ref, 'k--', label='ω = k', alpha=0.4)
        
        plt.xlabel('k (normalized to Ωp/VA)')
        plt.ylabel('ω (normalized to Ωp)')
        plt.title('Dispersion Relation Check\nAlfvén-Cyclotron Waves')
        plt.legend()
        plt.grid(True)
        
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'./figures/dispersion_check_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

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
            if self.dispersion_formula_choice == 'WangCB2014':
                self.k_z[i] = omega * (self.params['va2c'] **2 + 1/(1 - omega**2))**0.5
            elif self.dispersion_formula_choice == 'TuMarsch2001':
                self.k_z[i] = (omega**2 / (1-omega))**0.5
            elif self.dispersion_formula_choice == 'Cranmer2014':
                self.k_z[i] = (omega**2 / (1-omega))**0.5
            else:
                raise ValueError("Invalid dispersion formula choice. Choose from 'WangCB2014', 'TuMarsch2001', or 'Cranmer2014'.")


        # Initialize wave amplitudes with power-law spectrum
        alpha = 5/3  # Spectral index
        self.B_j = np.zeros(self.params['N'])
        
        # Calculate unnormalized amplitudes
        B_j_unnormalized = self.omega**(-alpha/2)
        
        # Normalize to maintain total energy
        total_energy = self.params['epsilon_0']
        normalization_factor = np.sqrt(2 * total_energy / np.sum(B_j_unnormalized**2))
        self.B_j = normalization_factor * B_j_unnormalized
        
        # Plot dispersion relation check
        self.plot_dispersion_check()

    def run_simulation(self, batch_size=500):
        """Run simulation for multiple particles using the C++ implementation with batch processing and streaming"""
        start_time = time.time()
        
        print("Starting multi-particle simulation with batch processing and real-time streaming...")
        
        # Create HDF5 file for streaming results
        base_dir = './data/solar_wind_ion_v2'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, base_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        h5_filename = os.path.join(save_dir, "simulation.h5")
        
        # Calculate total steps and create time array
        n_steps = int(self.params['simulation_time'] / self.params['dt'])
        t = np.arange(0, self.params['simulation_time'], self.params['dt'])
        
        # Initialize HDF5 file with metadata and structure
        with h5py.File(h5_filename, 'w') as f:
            # Save metadata
            f.attrs['creation_date'] = datetime.datetime.now().isoformat()
            f.attrs['description'] = 'Solar wind ion simulation results'
            f.attrs['simulation_status'] = 'in_progress'
            
            # Save simulation parameters
            param_group = f.create_group('simulation_parameters')
            for key, value in self.params.items():
                if isinstance(value, (list, np.ndarray)):
                    param_group.create_dataset(key, data=value)
                else:
                    param_group.attrs[key] = value
            
            # Save wave parameters
            wave_group = f.create_group('wave_parameters')
            wave_group.create_dataset('omega', data=self.omega)
            wave_group.create_dataset('k_z', data=self.k_z)
            wave_group.create_dataset('B_j', data=self.B_j)
            
            # Save time array
            f.create_dataset('time', data=t)
            
            # Create extendable dataset for trajectories
            trajectories = f.create_dataset('trajectories', 
                                          shape=(0, n_steps, 4),  # Start with 0 particles
                                          maxshape=(self.params['n_particles'], n_steps, 4),  # Allow expansion
                                          dtype='float64',
                                          chunks=(min(batch_size, 100), n_steps, 4),  # Optimize chunk size
                                          compression='gzip',
                                          compression_opts=6)
            
            # Create progress tracking dataset
            f.create_dataset('completed_particles', data=0)
        
        # Process particles in batches
        completed_particles = 0
        for batch_start in range(0, self.params['n_particles'], batch_size):
            batch_end = min(batch_start + batch_size, self.params['n_particles'])
            current_batch_size = batch_end - batch_start
            
            # Compute trajectories for current batch
            batch_solutions = self.cpp_core.run_multi_particle_simulation(
                current_batch_size,
                self.params['z_range'][0],
                self.params['z_range'][1],
                self.params['simulation_time'],
                batch_size
            )
            
            # Save batch results and update progress
            with h5py.File(h5_filename, 'a') as f:
                # Resize dataset to accommodate new particles
                current_size = f['trajectories'].shape[0]
                f['trajectories'].resize(current_size + current_batch_size, axis=0)
                
                # Write new batch data
                f['trajectories'][current_size:current_size + current_batch_size] = batch_solutions
                
                # Update progress
                completed_particles += current_batch_size
                f['completed_particles'][()] = completed_particles
                
                # Force write to disk
                f.flush()
            
            # Print progress
            elapsed_time = time.time() - start_time
            percentage = (completed_particles / self.params['n_particles']) * 100
            print(f"Progress: {percentage:.1f}% ({completed_particles}/{self.params['n_particles']} particles)")
            print(f"Batch {batch_start // batch_size + 1} completed in {elapsed_time:.2f} seconds")
            print(f"Results saved to: {h5_filename}")
        
        # Update file status to completed
        with h5py.File(h5_filename, 'a') as f:
            f.attrs['simulation_status'] = 'completed'
            f.attrs['completion_time'] = datetime.datetime.now().isoformat()
            f.attrs['total_runtime'] = time.time() - start_time
        
        total_elapsed_time = time.time() - start_time
        print(f"Simulation completed in {total_elapsed_time:.2f} seconds")
        
        return t, h5_filename

    def load_results(self, h5_filename, sampling_rate=1, particle_range=None, time_range=None, chunk_size=1000):
        """
        Load simulation results from HDF5 file with options for partial loading and chunking
        
        Parameters:
        -----------
        h5_filename : str
            Path to the HDF5 file containing simulation results
        sampling_rate : int, optional
            Load every Nth timestep to reduce memory usage. Default is 1 (load all steps)
        particle_range : tuple, optional
            Range of particles to load (start, end). Default is None (load all particles)
        time_range : tuple, optional
            Range of time steps to load (start, end). Default is None (load all time steps)
        chunk_size : int, optional
            Number of particles to load in each chunk. Default is 1000
        
        Returns:
        --------
        t : numpy.ndarray
            Time array
        solutions_generator : generator
            Generator that yields chunks of the simulation results
        """
        with h5py.File(h5_filename, 'r') as f:
            # Load time array
            t = np.arange(0, self.params['simulation_time'], self.params['dt'])
            if time_range:
                t = t[time_range[0]:time_range[1]:sampling_rate]
            else:
                t = t[::sampling_rate]
            
            # Set up particle range
            if particle_range is None:
                particle_range = (0, f['trajectories'].shape[0])
            
            # Set up time range for data loading
            if time_range is None:
                time_slice = slice(None, None, sampling_rate)
            else:
                time_slice = slice(time_range[0], time_range[1], sampling_rate)
            
            # Create a generator to yield data chunks
            def solutions_generator():
                for chunk_start in range(particle_range[0], particle_range[1], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, particle_range[1])
                    yield f['trajectories'][chunk_start:chunk_end, time_slice, :]
        
        return t, solutions_generator()

    def plot_velocity_distribution(self, h5_filename, times, sampling_rate=1, particle_range=None):
        """Plot velocity distribution at different times"""
        fig, axs = plt.subplots(2, len(times), figsize=(5*len(times), 10))
        fig.suptitle('Ion Velocity Distribution at Different Times')
        
        t, solutions_gen = self.load_results(h5_filename, sampling_rate, particle_range)
        
        for i, target_time in enumerate(times):
            time_index = np.argmin(np.abs(t - target_time))
            
            vx, vy, vz = [], [], []
            for chunk in solutions_gen:
                vx.extend(chunk[:, time_index, 1])
                vy.extend(chunk[:, time_index, 2])
                vz.extend(chunk[:, time_index, 3])
            
            vx = np.array(vx)
            vy = np.array(vy)
            vz = np.array(vz)
            v_perp = np.sqrt(vx**2 + vy**2)
            
            # vx-vz plane
            axs[0, i].hist2d(vz, vx, bins=50, cmap='jet')
            axs[0, i].set_xlabel('vz')
            axs[0, i].set_ylabel('vx')
            axs[0, i].set_title(f't = {target_time:.2f}')
            
            # v_perp-vz plane
            axs[1, i].hist2d(vz, v_perp, bins=50, cmap='jet')
            axs[1, i].set_xlabel('vz')
            axs[1, i].set_ylabel('v_perp')
        
        plt.tight_layout()
        plt.savefig(f'./figures/velocity_distribution_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()

    def save_to_hdf5(self, h5_filename, simulation_type='normal_mode'):
        """
        Update HDF5 file with additional metadata
        
        Parameters:
        -----------
        h5_filename : str
            Path to the existing HDF5 file containing simulation results
        simulation_type : str, optional
            Type of simulation for organization
        """
        with h5py.File(h5_filename, 'r+') as f:
            f.attrs['simulation_type'] = simulation_type
            f.attrs['last_modified'] = datetime.datetime.now().isoformat()
        
        log_print(f"Simulation metadata updated in {h5_filename}")
        return h5_filename

    def plot_velocity_distribution(self, all_solutions, times):
        """Plot velocity distribution at different times"""
        fig, axs = plt.subplots(2, len(times), figsize=(5*len(times), 10))
        fig.suptitle('Ion Velocity Distribution at Different Times')
        
        # 重定向标准输出和标准错误到run.log
        sys.stdout = open('run.log', 'a')
        sys.stderr = sys.stdout
        
        for i, time in enumerate(times):
            time_index = int(time / self.params['dt'])
            if time_index >= all_solutions.shape[1]:
                print(f"Warning: Time {time} is beyond simulation time")
                continue
                
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
        plt.savefig('./figures/velocity_distribution.png')
        plt.close()
        
        # 恢复标准输出和标准错误
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        log_print("Velocity distribution plot saved to ./figures/velocity_distribution.png")

    def plot_particle_trajectories(self, t, solutions, num_particles=5):
        """Plot trajectories for selected particles"""
        selected_indices = np.random.choice(solutions.shape[0], num_particles, replace=False)
        
        fig = plt.figure(figsize=(15, 5))
        
        # 重定向标准输出和标准错误到run.log
        sys.stdout = open('run.log', 'a')
        sys.stderr = sys.stdout
        
        # 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        for idx in selected_indices:
            ax1.plot(solutions[idx, :, 1], solutions[idx, :, 2], solutions[idx, :, 0])
        ax1.set_xlabel('vx (VA)')
        ax1.set_ylabel('vy (VA)')
        ax1.set_zlabel('z (VA/Ωp)')
        ax1.set_title('Particle Trajectories')
        
        # Velocity components
        ax2 = fig.add_subplot(132)
        for idx in selected_indices:
            ax2.plot(t, solutions[idx, :, 1], label=f'vx_{idx}')
            ax2.plot(t, solutions[idx, :, 2], label=f'vy_{idx}')
            ax2.plot(t, solutions[idx, :, 3], label=f'vz_{idx}')
        ax2.set_xlabel('Time (Ωp^-1)')
        ax2.set_ylabel('Velocity (VA)')
        ax2.set_title('Velocity Components')
        ax2.legend()
        
        # Magnetic moment
        ax3 = fig.add_subplot(133)
        for idx in selected_indices:
            v_perp_squared = solutions[idx, :, 1]**2 + solutions[idx, :, 2]**2
            magnetic_moment = 0.5 * v_perp_squared
            ax3.plot(t, magnetic_moment, label=f'Particle {idx}')
        ax3.set_xlabel('Time (Ωp^-1)')
        ax3.set_ylabel('Magnetic Moment (normalized)')
        ax3.set_title('Magnetic Moment')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('./figures/particle_trajectories.png')
        plt.close()
        
        # 恢复标准输出和标准错误
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        log_print("Particle trajectories plot saved to ./figures/particle_trajectories.png")


def check_dispersion_relation(beta=0.1):
    """
    独立函数，用于检查和绘制阿尔芬-离子回旋波的色散关系
    
    Parameters:
    beta : float
        等离子体 beta 值
    """
    plt.figure(figsize=(10, 8))
    
    # 计算连续色散关系
    k_values = np.linspace(0.01, 5, 1000)
    omega_alfven = k_values / np.sqrt(1 + beta/2)  # 阿尔芬波色散关系
    
    # 计算离子回旋波色散关系
    omega_ic = np.zeros_like(k_values)
    for i, k in enumerate(k_values):
        # 使用二分法求解离子回旋波色散关系
        omega_min, omega_max = 0.001, 0.999
        while omega_max - omega_min > 1e-6:
            omega = (omega_min + omega_max) / 2
            k_test = omega * np.sqrt(1 + 1/(1 - omega**2))
            if k_test < k:
                omega_min = omega
            else:
                omega_max = omega
        omega_ic[i] = (omega_min + omega_max) / 2
    
    # 绘制色散曲线
    plt.plot(k_values, omega_alfven, 'b-', label='Alfvén Wave (β = {})'.format(beta))
    plt.plot(k_values, omega_ic, 'r-', label='Ion Cyclotron Wave')
    plt.plot(k_values, k_values, 'k--', label='ω = k', alpha=0.4)
    plt.axhline(y=1.0, color='g', linestyle='--', label='Ion Cyclotron Frequency')
    
    plt.xlabel('k (normalized to VA/Ωp)')
    plt.ylabel('ω (normalized to Ωp)')
    plt.title('Dispersion Relation for Alfvén-Cyclotron Waves')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 5)
    plt.ylim(0, 2)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'./figures/dispersion_relation_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    log_print(f"色散关系图已保存到 ./figures/dispersion_relation_{timestamp}.png")

if __name__ == "__main__":
    # 清空run.log文件
    open('run.log', 'w').close()
    
    # 创建模拟实例，使用默认参数
    sim = SolarWindIonSimulationV2()
    
    # 可以根据需要修改特定参数
    #a sim.params['simulation_time'] = 1000  # 修改模拟总时间
    #a sim.params['n_particles'] = 2000  # 设置粒子数量
    
    # 重新初始化波参数，因为我们修改了参数
    log_print("初始化波参数并检查色散关系...")
    sim.initialize_waves()
    # 显式调用色散关系检查，并保存图像
    sim.plot_dispersion_check(show=False, save=True)
    log_print("色散关系检查完成，图像已保存到 ./figures/dispersion_check_*.png")
    
    # 重新初始化C++核心
    sim.cpp_core = SolarWindCoreV2(
        sim.omega,
        sim.k_z,
        sim.B_j,
        sim.params['epsilon_0'],
        sim.params['tau'],
        sim.params['delta_tau'],
        sim.params['dt'],
        sim.params['beta']
    )
    
    # 设置模拟类型和批处理大小
    simulation_type = 'normal_mode'  # 可选: 'normal_mode', 'wave_amplitude_study', 'beta_scan', 'long_time'
    batch_size = 500  # 每批处理的粒子数量
    sampling_rate = 10  # 每10个时间步加载一次数据，减小内存使用
    
    # Run simulation with batch processing and streaming
    log_print("Starting simulation with batch processing and streaming...")
    start_time = time.time()
    t, h5_filename = sim.run_simulation(batch_size)
    end_time = time.time()
    log_print(f"Simulation completed! Total time: {end_time - start_time:.2f} seconds")
    
    # Update HDF5 file with additional metadata
    log_print("Updating HDF5 file metadata...")
    h5_file_path = sim.save_to_hdf5(h5_filename, simulation_type=simulation_type)
    
    # Load results for plotting
    log_print("Loading results for plotting...")
    t, all_solutions = sim.load_results(h5_file_path, sampling_rate=sampling_rate)
    
    # Plot results
    log_print("Plotting results...")
    
    # Plot velocity distributions at different times
    times_to_plot = [0, 500, 1000]
    sim.plot_velocity_distribution(all_solutions, times_to_plot)
    
    # Plot trajectories for a few particles
    sim.plot_particle_trajectories(t, all_solutions, num_particles=5)
    
    log_print("All plots have been saved in the ./figures directory.")
    log_print(f"Simulation data saved to {h5_file_path}")