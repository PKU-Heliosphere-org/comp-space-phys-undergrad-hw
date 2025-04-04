#!/usr/bin/env python3
"""
测试脚本：运行优化后的太阳风离子模拟并绘制结果
"""
import os
import sys
import matplotlib.pyplot as plt
from solar_wind_ion_simulation_v2 import SolarWindIonSimulationV2

def main():
    # 创建模拟实例
    sim = SolarWindIonSimulationV2()

    # 使用较小的参数进行测试
    test_params = {
        'simulation_time': 1000,  # 减少模拟时间
        'n_particles': 500,      # 减少粒子数
        'output_sampling': 10,   # 每10步保存一次
        'dt': 0.1                # 增加时间步长
    }
    sim.params.update(test_params)

    # 确保输出目录存在
    os.makedirs('./figures', exist_ok=True)
    
    # 运行模拟
    print('Starting test simulation...')
    t, h5_file = sim.run_simulation('test_simulation.h5')
    print(f'Simulation completed, results saved to {h5_file}')

    # 绘制结果
    print('Plotting results...')
    sim.plot_velocity_distribution(
        'test_simulation.h5',
        times=[100, 500],  # 只看两个时间点
        sampling_rate=1,
        particle_range=(0, 500)  # 只看前500个粒子
    )
    
    # 打印图片保存位置
    print('Done! Check the figures directory for output plots.')

if __name__ == "__main__":
    main()