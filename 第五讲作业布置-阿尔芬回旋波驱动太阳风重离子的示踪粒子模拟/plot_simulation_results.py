#!/usr/bin/env python3
"""
专门用于加载和绘制太阳风离子模拟结果的程序
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class SimulationPlotter:
    def __init__(self):
        """初始化绘图器"""
        # 确保输出目录存在
        os.makedirs('./figures', exist_ok=True)
    
    def get_file_info(self, h5_filename):
        """获取HDF5文件的基本信息"""
        with h5py.File(h5_filename, 'r') as f:
            # 获取数据集的形状
            shape = f['trajectories'].shape
            # 获取文件中保存的参数
            params = dict(f.attrs)
            
            print("文件信息:")
            print(f"粒子数: {shape[0]}")
            print(f"时间步数: {shape[1]}")
            print(f"每个点的维度: {shape[2]}")
            print("\n模拟参数:")
            for key, value in params.items():
                print(f"{key}: {value}")
            
            return shape, params
    
    def plot_velocity_distribution(self, h5_filename, times, particle_range=None, chunk_size=1000):
        """
        绘制速度分布
        
        参数:
        h5_filename: str, HDF5文件路径
        times: list, 要绘制的时间点列表
        particle_range: tuple, 可选，要分析的粒子范围 (start, end)
        chunk_size: int, 可选，每次读取的粒子数量
        """
        print(f"开始绘制速度分布...")
    
        if chunk_size > 10000:
            print("警告：chunk_size过大，已自动调整为10000")
            chunk_size = 10000
    
        with h5py.File(h5_filename, 'r') as f:
            # 获取数据集信息
            shape = f['trajectories'].shape
            dt = f.attrs.get('dt', 0.05)  # 默认dt=0.05
            output_sampling = f.attrs.get('output_sampling', 1)  # 默认output_sampling=1
            
            # 计算实际的时间数组
            dt_effective = dt * output_sampling
            t = np.arange(0, shape[1]) * dt_effective
            
            # 设置粒子范围，并确保不会超过数据集大小
            if particle_range is None:
                particle_range = (0, min(shape[0], 100000))  # 限制最大粒子数为100000
            else:
                particle_range = (min(particle_range[0], shape[0]), min(particle_range[1], shape[0]))
                if particle_range[1] - particle_range[0] > 100000:
                    print("警告：选择的粒子数量过多，已自动限制为前100000个粒子")
                    particle_range = (particle_range[0], particle_range[0] + 100000)
            
            print(f"分析粒子范围: {particle_range[0]} 到 {particle_range[1]}")
            
            # 初始化用于存储所有速度数据的列表
            all_vx = []
            all_vy = []
            all_vz = []
            
            # 遍历所有时间点，收集速度数据
            for target_time in times:
                time_index = np.argmin(np.abs(t - target_time))
                
                vx, vy, vz = [], [], []
                
                for chunk_start in range(particle_range[0], particle_range[1], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, particle_range[1])
                    chunk = f['trajectories'][chunk_start:chunk_end, time_index, :]
                    
                    vx.extend(chunk[:, 1])
                    vy.extend(chunk[:, 2])
                    vz.extend(chunk[:, 3])
                
                all_vx.extend(vx)
                all_vy.extend(vy)
                all_vz.extend(vz)
            
            # 转换为numpy数组
            all_vx = np.array(all_vx)
            all_vy = np.array(all_vy)
            all_vz = np.array(all_vz)
            
            # 计算统一的速度范围（使用分位数）
            v_min = np.percentile(np.hstack((all_vx, all_vy, all_vz)), 1)
            v_max = np.percentile(np.hstack((all_vx, all_vy, all_vz)), 99)
            v_range = (v_min, v_max)
            
            # 创建图形
            fig, axs = plt.subplots(2, len(times), figsize=(4*len(times), 8))
            if len(times) == 1:
                axs = np.array([[axs[0]], [axs[1]]])
            fig.suptitle('Ion Velocity Distribution at Different Times')
            
            # 对每个时间点进行处理并绘制
            for i, target_time in enumerate(times):
                time_index = np.argmin(np.abs(t - target_time))
                
                vx, vy, vz = [], [], []
                
                for chunk_start in range(particle_range[0], particle_range[1], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, particle_range[1])
                    chunk = f['trajectories'][chunk_start:chunk_end, time_index, :]
                    
                    vx.extend(chunk[:, 1])
                    vy.extend(chunk[:, 2])
                    vz.extend(chunk[:, 3])
                
                vx = np.array(vx)
                vy = np.array(vy)
                vz = np.array(vz)
                v_perp = np.sqrt(vx**2 + vy**2)
                
                # 绘制vx-vz平面的分布
                h1 = axs[0, i].hist2d(vz, vx, bins=50, cmap='jet', range=[v_range, v_range])
                axs[0, i].set_xlabel('vz')
                axs[0, i].set_ylabel('vx')
                axs[0, i].set_title(f't = {target_time:.2f}')
                axs[0, i].set_aspect('equal')
                plt.colorbar(h1[3], ax=axs[0, i])
                
                # 绘制v_perp-vz平面的分布
                h2 = axs[1, i].hist2d(vz, v_perp, bins=50, cmap='jet', range=[v_range, (0, v_max)])
                axs[1, i].set_xlabel('vz')
                axs[1, i].set_ylabel('v_perp')
                axs[1, i].set_aspect('equal')
                plt.colorbar(h2[3], ax=axs[1, i])
                
                # 设置统一的横纵坐标范围
                axs[0, i].set_xlim(v_range)
                axs[0, i].set_ylim(v_range)
                axs[1, i].set_xlim(v_range)
                axs[1, i].set_ylim((0, v_max))
                
                # 添加网格线
                axs[0, i].grid(True, linestyle='--', alpha=0.3)
                axs[1, i].grid(True, linestyle='--', alpha=0.3)
            
            # 调整布局并保存
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = f'./figures/velocity_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(save_path)
            plt.close()
            
            print(f"图形已保存至: {save_path}")
    
    def plot_trajectories(self, h5_filename, particle_indices, time_range=None):
        """
        绘制选定粒子的轨迹
        
        参数:
        h5_filename: str, HDF5文件路径
        particle_indices: list, 要绘制的粒子索引列表
        time_range: tuple, 可选，要绘制的时间范围 (start, end)
        """
        print(f"开始绘制粒子轨迹...")
        
        with h5py.File(h5_filename, 'r') as f:
            # 获取时间信息
            dt = f.attrs.get('dt', 0.05)
            output_sampling = f.attrs.get('output_sampling', 1)
            dt_effective = dt * output_sampling
            
            # 选择时间范围
            total_timesteps = f['trajectories'].shape[1]
            if time_range is None:
                time_slice = slice(None)
                t = np.arange(total_timesteps) * dt_effective
            else:
                start_idx = min(int(time_range[0] / dt_effective), total_timesteps - 1)
                end_idx = min(int(time_range[1] / dt_effective), total_timesteps)
                time_slice = slice(start_idx, end_idx)
                t = np.arange(start_idx, end_idx) * dt_effective
            
            # 创建图形
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            fig.suptitle('Particle Trajectories')
            
            # 为每个选定的粒子绘制轨迹
            for idx in particle_indices:
                if idx >= f['trajectories'].shape[0]:
                    print(f"警告：粒子索引 {idx} 超出范围，已跳过")
                    continue
                
                trajectory = f['trajectories'][idx, time_slice, :]
                
                # 绘制z位置随时间的变化
                axs[0].plot(t, trajectory[:, 0], label=f'Particle {idx}')
                
                # 绘制速度分量随时间的变化
                axs[1].plot(t, trajectory[:, 1], label=f'vx {idx}')
                axs[1].plot(t, trajectory[:, 2], label=f'vy {idx}')
                axs[1].plot(t, trajectory[:, 3], label=f'vz {idx}')
            
            # 设置标签和图例
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('z position')
            axs[0].legend()
            
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Velocity components')
            axs[1].legend()
            
            # 保存图形
            plt.tight_layout()
            save_path = f'./figures/trajectories_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(save_path)
            plt.close()
            
            print(f"轨迹图已保存至: {save_path}")


def list_h5_files(directory):
    """列出指定目录下的所有simulation.h5文件及其路径"""
    h5_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == 'simulation.h5':
                h5_files.append(os.path.join(root, file))
    return h5_files

def main():
    """主函数"""
    # 创建绘图器实例
    plotter = SimulationPlotter()
    
    # 设置要分析的HDF5文件路径
    # h5_dir = '/Users/jshept/Documents/GitHubOrg/From_StackEdit/本科生课程讲义/第五讲讲义/第五讲作业布置/data/solar_wind_ion_v2/'
    # h5_file = h5_dir + '/20250404_122345/simulation.h5'
    
    # 检查文件是否存在
    # if not os.path.exists(h5_file):
    #     print(f"错误：文件 {h5_file} 不存在。")
    #     return

    # 设置要分析的HDF5文件目录
    h5_dir = '/Users/jshept/Documents/GitHubOrg/From_StackEdit/本科生课程讲义/第五讲讲义/第五讲作业布置/data/solar_wind_ion_v2/'
    
    # 列出所有simulation.h5文件
    h5_files = list_h5_files(h5_dir)
    
    if not h5_files:
        print("错误：未找到任何simulation.h5文件。")
        return
    
    # 打印文件列表并让用户选择
    print("找到以下simulation.h5文件：")
    for i, h5_file in enumerate(h5_files, 1):
        print(f"{i}. {h5_file}")
    
    choice = input("请输入要分析的文件编号：")
    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(h5_files):
            raise ValueError
        h5_file = h5_files[choice_index]
    except ValueError:
        print("错误：无效的输入。")
        return
    
    # 检查文件是否存在（理论上应该存在，因为我们是从列表中选择的）
    if not os.path.exists(h5_file):
        print(f"错误：文件 {h5_file} 不存在。")
        return

    try:
        # 获取并显示文件信息
        shape, params = plotter.get_file_info(h5_file)
        
        # 根据文件信息确定合适的分析参数
        total_particles = shape[0]
        total_timesteps = shape[1]
        dt = params.get('dt', 0.05)
        output_sampling = params.get('output_sampling', 1)
        
        # 计算实际时间点
        dt_effective = dt * output_sampling
        actual_times = np.arange(total_timesteps) * dt_effective
        
        # 选择三个时间点：开始、中间和结束
        time_indices = [0, total_timesteps // 2, total_timesteps - 1]
        times_to_plot = [actual_times[i] for i in time_indices]
        
        print(f"\n选择的时间点：")
        for t in times_to_plot:
            print(f"t = {t:.2f}")
        
        # 绘制速度分布
        plotter.plot_velocity_distribution(
            h5_file,
            times=times_to_plot,
            particle_range=(0, min(total_particles, 10000)),  # 分析前10000个粒子或全部粒子
            chunk_size=1000  # 每次读取1000个粒子
        )
        
        # 绘制部分粒子的轨迹
        particle_indices = [0, total_particles // 4, total_particles // 2, 3 * total_particles // 4, total_particles - 1]
        particle_indices = [idx for idx in particle_indices if idx < total_particles]
        plotter.plot_trajectories(
            h5_file,
            particle_indices=particle_indices,  # 选择5个或更少的分布均匀的粒子
            time_range=(0, actual_times[-1])  # 绘制整个时间范围的轨迹
        )
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()