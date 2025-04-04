import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import root_scalar
import matplotlib as mpl

# 设置更好的图形风格
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

def alfven_dispersion(k, beta_i=0.1, beta_e=0.1):
    """
    计算阿尔芬波的色散关系
    简化版本: ω = k / sqrt(1 + beta/2)
    
    参数:
    k : float 或 array
        波数 (归一化到离子惯性长度)
    beta_i : float
        离子等离子体 beta 值
    beta_e : float
        电子等离子体 beta 值
        
    返回:
    float 或 array
        频率 (归一化到离子回旋频率)
    """
    beta = beta_i + beta_e
    return k / np.sqrt(1 + beta/2)

def ion_cyclotron_dispersion_equation(omega, k, beta_i=0.1, beta_e=0.1):
    """
    离子回旋波的色散关系方程
    
    参数:
    omega : float
        频率 (归一化到离子回旋频率)
    k : float
        波数 (归一化到离子惯性长度)
    beta_i : float
        离子等离子体 beta 值
    beta_e : float
        电子等离子体 beta 值
        
    返回:
    float
        色散关系方程值 (为零时表示解)
    """
    # 离子项
    ion_term = (1 - omega) / (1 - omega**2)
    
    # 电子项 (假设电子质量远小于离子质量)
    electron_term = beta_e / (2 * (1 + beta_e/2))
    
    # 完整色散关系
    return omega**2 - k**2 * (1 - ion_term * beta_i/2 - electron_term)

def find_ion_cyclotron_omega(k, beta_i=0.1, beta_e=0.1, bracket=None):
    """
    对于给定的波数，求解离子回旋波的频率
    
    参数:
    k : float
        波数 (归一化到离子惯性长度)
    beta_i : float
        离子等离子体 beta 值
    beta_e : float
        电子等离子体 beta 值
    bracket : tuple, optional
        求解区间 (min_omega, max_omega)
        
    返回:
    float
        频率 (归一化到离子回旋频率)
    """
    if k == 0:
        return 0
    
    # 设置求解区间
    if bracket is None:
        min_omega = 0.01
        max_omega = min(0.99, 0.95 * alfven_dispersion(k, beta_i, beta_e))
        bracket = (min_omega, max_omega)
    
    try:
        # 使用根求解器找到色散关系的解
        result = root_scalar(
            lambda w: ion_cyclotron_dispersion_equation(w, k, beta_i, beta_e),
            bracket=bracket,
            method='brentq'
        )
        return result.root
    except:
        # 如果求解失败，返回 NaN
        return np.nan

def calculate_dispersion_curves(k_values, beta_i_values, beta_e=0.1):
    """
    计算不同 beta_i 值的色散曲线
    
    参数:
    k_values : array
        波数值数组
    beta_i_values : list
        离子等离子体 beta 值列表
    beta_e : float
        电子等离子体 beta 值
        
    返回:
    dict
        包含阿尔芬波和离子回旋波色散曲线的字典
    """
    results = {
        'k': k_values,
        'alfven': {},
        'ion_cyclotron': {}
    }
    
    for beta_i in beta_i_values:
        # 计算阿尔芬波色散关系
        results['alfven'][beta_i] = alfven_dispersion(k_values, beta_i, beta_e)
        
        # 计算离子回旋波色散关系
        ic_omega = []
        for k in k_values:
            omega = find_ion_cyclotron_omega(k, beta_i, beta_e)
            ic_omega.append(omega)
        
        results['ion_cyclotron'][beta_i] = np.array(ic_omega)
    
    return results

def plot_dispersion_curves(results, beta_i_values, beta_e, save_path='dispersion_relations.png'):
    """
    绘制色散曲线
    
    参数:
    results : dict
        由 calculate_dispersion_curves 返回的结果
    beta_i_values : list
        离子等离子体 beta 值列表
    beta_e : float
        电子等离子体 beta 值
    save_path : str
        保存图像的路径
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, height_ratios=[2, 1])
    
    # 主色散图
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0, 1, len(beta_i_values)))
    
    for i, beta_i in enumerate(beta_i_values):
        color = colors[i]
        # 绘制阿尔芬波
        ax1.plot(results['k'], results['alfven'][beta_i], 
                 color=color, linestyle='-', 
                 label=f'Alfvén (βi = {beta_i})')
        
        # 绘制离子回旋波
        ax1.plot(results['k'], results['ion_cyclotron'][beta_i], 
                 color=color, linestyle='--', 
                 label=f'Ion Cyclotron (βi = {beta_i})')
    
    # 绘制 ω = 1 的水平线 (离子回旋频率)
    ax1.axhline(y=1.0, color='r', linestyle='-', alpha=0.5, 
                label='Ion Cyclotron Frequency (ω = 1)')
    
    ax1.set_xlabel('k (normalized to ion inertial length)')
    ax1.set_ylabel('ω (normalized to ion cyclotron frequency)')
    ax1.set_title('Dispersion Relations for Alfvén and Ion Cyclotron Waves')
    ax1.set_xlim(0, results['k'].max())
    ax1.set_ylim(0, 2.0)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax1.grid(True)
    
    # 相速度图
    ax2 = fig.add_subplot(gs[1, 0])
    for i, beta_i in enumerate(beta_i_values):
        color = colors[i]
        # 计算并绘制阿尔芬波相速度
        vph_alfven = results['alfven'][beta_i] / results['k']
        vph_alfven[0] = vph_alfven[1]  # 避免 k=0 时的除零
        ax2.plot(results['k'], vph_alfven, 
                 color=color, linestyle='-',
                 label=f'Alfvén (βi = {beta_i})')
        
        # 计算并绘制离子回旋波相速度
        vph_ic = results['ion_cyclotron'][beta_i] / results['k']
        vph_ic[0] = vph_ic[1]  # 避免 k=0 时的除零
        ax2.plot(results['k'], vph_ic, 
                 color=color, linestyle='--',
                 label=f'Ion Cyclotron (βi = {beta_i})')
    
    ax2.set_xlabel('k (normalized to ion inertial length)')
    ax2.set_ylabel('Phase Velocity (vA)')
    ax2.set_title('Phase Velocity vs. Wave Number')
    ax2.set_xlim(0, results['k'].max())
    ax2.set_ylim(0, 1.5)
    ax2.grid(True)
    
    # 群速度图
    ax3 = fig.add_subplot(gs[1, 1])
    for i, beta_i in enumerate(beta_i_values):
        color = colors[i]
        # 计算并绘制阿尔芬波群速度 (dω/dk)
        dw_dk_alfven = np.gradient(results['alfven'][beta_i], results['k'])
        ax3.plot(results['k'], dw_dk_alfven, 
                 color=color, linestyle='-',
                 label=f'Alfvén (βi = {beta_i})')
        
        # 计算并绘制离子回旋波群速度
        dw_dk_ic = np.gradient(results['ion_cyclotron'][beta_i], results['k'])
        ax3.plot(results['k'], dw_dk_ic, 
                 color=color, linestyle='--',
                 label=f'Ion Cyclotron (βi = {beta_i})')
    
    ax3.set_xlabel('k (normalized to ion inertial length)')
    ax3.set_ylabel('Group Velocity (vA)')
    ax3.set_title('Group Velocity vs. Wave Number')
    ax3.set_xlim(0, results['k'].max())
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dispersion relation plots have been saved to '{save_path}'")

def main():
    # 设置参数
    k_max = 5.0
    k_values = np.linspace(0, k_max, 500)
    beta_i_values = [0.01, 0.1, 0.5, 1.0]
    beta_e = 0.1
    
    print("Calculating dispersion relations...")
    results = calculate_dispersion_curves(k_values, beta_i_values, beta_e)
    
    print("Plotting dispersion curves...")
    plot_dispersion_curves(results, beta_i_values, beta_e, 'dispersion_relations.png')
    
    # 打印一些关键值
    print("\nKey values of the dispersion relation:")
    print("----------------------------------------")
    print("k\tω (Alfvén, βi=0.1)\tω (Ion Cyclotron, βi=0.1)")
    for i, k in enumerate(k_values):
        if i % 50 == 0:  # 只打印一部分值
            print(f"{k:.2f}\t{results['alfven'][0.1][i]:.4f}\t\t{results['ion_cyclotron'][0.1][i]:.4f}")

if __name__ == "__main__":
    main()