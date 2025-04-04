#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <array>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>

namespace py = pybind11;

// 定义常量 M_PI（如果系统没有定义）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SolarWindCoreV2 {
private:
    std::vector<double> m_omega, m_k_z, m_B_j;
    double m_epsilon_0, m_tau, m_delta_tau;
    double m_dt;
    int m_N;
    double m_beta;  // 添加热速度参数

    double wave_amplitude_time_dependence(double t) const {
        if (t < m_tau) {
            return m_epsilon_0 * std::exp(-std::pow(t - m_tau, 2) / std::pow(m_delta_tau, 2));
        }
        return m_epsilon_0;
    }

    std::tuple<double, double, double, double> calculate_fields(double z, double t) const {
        double Bx = 0.0, By = 0.0, Ex = 0.0, Ey = 0.0;
        const double time_amplitude = wave_amplitude_time_dependence(t);
        
        for (int j = 0; j < m_N; ++j) {
            const double phase = m_k_z[j] * z - m_omega[j] * t;
            const double amplitude = m_B_j[j] * time_amplitude;
            
            // Left-hand circular polarization in x-y plane
            // Modification for By-component to match the left-hand circular polarization
            Bx += +amplitude * std::cos(phase);
            By += -amplitude * std::sin(phase);
            
            // Electric field components
            // Note: The minus sign is used to match the left-hand circular polarization
            const double E_amplitude = amplitude * (m_omega[j] / m_k_z[j]);
            Ex += -E_amplitude * std::sin(phase);
            Ey += -E_amplitude * std::cos(phase);
        }
        
        return {Ex, Ey, Bx, By};
    }

    std::tuple<double, std::array<double, 3>> boris_push(
        double z, const std::array<double, 3>& v, double t, double dt) {
        
        // Constants for O5+ ion (normalized to proton values)
        const double q_m_ratio = 5.0/16.0;  // Charge-to-mass ratio for O5+
        
        // Get fields at current position
        auto [Ex, Ey, Bx, By] = calculate_fields(z, t);
        const double Bz = 1.0;  // Background field
        const double Ez = 0.0;  // No parallel electric field
        
        // Field vectors
        const std::array<double, 3> E = {Ex, Ey, Ez};
        const std::array<double, 3> B = {Bx, By, Bz};
        
        // 1. Half acceleration in E field
        std::array<double, 3> v_minus;
        for (int i = 0; i < 3; ++i) {
            v_minus[i] = v[i] + 0.5 * dt * q_m_ratio * E[i];
        }
        
        // 2. Magnetic field rotation
        std::array<double, 3> t_vec;
        for (int i = 0; i < 3; ++i) {
            t_vec[i] = 0.5 * dt * q_m_ratio * B[i];
        }
        
        double t_squared = 0;
        for (int i = 0; i < 3; ++i) {
            t_squared += t_vec[i] * t_vec[i];
        }
        
        // Boris rotation factors
        std::array<double, 3> s_vec;
        for (int i = 0; i < 3; ++i) {
            s_vec[i] = 2 * t_vec[i] / (1 + t_squared);
        }
        
        // Cross products for rotation
        std::array<double, 3> v_prime = {
            //original v_minus + cross product of t_vec and v_minus
            v_minus[0] + (v_minus[1] * t_vec[2] - v_minus[2] * t_vec[1]),
            v_minus[1] + (v_minus[2] * t_vec[0] - v_minus[0] * t_vec[2]),
            v_minus[2] + (v_minus[0] * t_vec[1] - v_minus[1] * t_vec[0])
        };
        
        std::array<double, 3> v_plus = v_minus;
        std::array<double, 3> v_cross_s = {
            v_prime[1] * s_vec[2] - v_prime[2] * s_vec[1],
            v_prime[2] * s_vec[0] - v_prime[0] * s_vec[2],
            v_prime[0] * s_vec[1] - v_prime[1] * s_vec[0]
        };
        
        for (int i = 0; i < 3; ++i) {
            v_plus[i] += v_cross_s[i];
        }
        
        // 3. Half acceleration in E field
        std::array<double, 3> new_v;
        for (int i = 0; i < 3; ++i) {
            new_v[i] = v_plus[i] + 0.5 * dt * q_m_ratio * E[i];
        }
        
        // 4. Position update
        // double new_z = z + 0.5 * dt * (v[2] + new_v[2]);
        double new_z = z + dt * new_v[2];
        
        return {new_z, new_v};
    }

public:
    SolarWindCoreV2(
        const py::array_t<double>& omega,
        const py::array_t<double>& k_z,
        const py::array_t<double>& B_j,
        double epsilon_0,
        double tau,
        double delta_tau,
        double dt,
        double beta
    ) : m_epsilon_0(epsilon_0), m_tau(tau), m_delta_tau(delta_tau),
        m_dt(dt), m_beta(beta) {
        
        auto omega_buf = omega.request();
        auto k_z_buf = k_z.request();
        auto B_j_buf = B_j.request();
        
        m_N = omega_buf.shape[0];
        
        m_omega.resize(m_N);
        m_k_z.resize(m_N);
        m_B_j.resize(m_N);
        
        std::memcpy(m_omega.data(), omega_buf.ptr, m_N * sizeof(double));
        std::memcpy(m_k_z.data(), k_z_buf.ptr, m_N * sizeof(double));
        std::memcpy(m_B_j.data(), B_j_buf.ptr, m_N * sizeof(double));
    }

    py::array_t<double> run_particle_simulation(
        double z_init,
        double vx_init,
        double vy_init,
        double vz_init,
        double total_time
    ) {
        int n_steps = static_cast<int>(total_time / m_dt);
        py::array_t<double> trajectory({n_steps, 4});  // [z, vx, vy, vz]
        auto buf = trajectory.mutable_unchecked<2>();
        
        double z = z_init;
        std::array<double, 3> v = {vx_init, vy_init, vz_init};
        
        // Store initial state
        buf(0, 0) = z;
        buf(0, 1) = v[0];
        buf(0, 2) = v[1];
        buf(0, 3) = v[2];
        
        // Time integration using Boris algorithm
        for (int i = 1; i < n_steps; ++i) {
            double t = i * m_dt;
            std::tie(z, v) = boris_push(z, v, t, m_dt);
            
            buf(i, 0) = z;
            buf(i, 1) = v[0];
            buf(i, 2) = v[1];
            buf(i, 3) = v[2];
        }
        
        return trajectory;
    }

    py::array_t<double> run_multi_particle_simulation(
        int n_particles,
        double z_min,
        double z_max,
        double total_time,
        int batch_size = 500  // 批处理大小参数
    ) {
        // 初始化随机数生成器
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        
        // Box-Muller 变换生成正态分布随机数
        auto normal_random = [](double mean, double stddev) -> double {
            double u1 = (std::rand() + 1.0) / (RAND_MAX + 1.0);
            double u2 = (std::rand() + 1.0) / (RAND_MAX + 1.0);
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            return mean + stddev * z;
        };

        int n_steps = static_cast<int>(total_time / m_dt);
        double v_std = std::sqrt(m_beta);
        
        // 创建输出数组，只为当前批次的粒子分配内存
        py::array_t<double> batch_trajectories({n_particles, n_steps, 4});
        auto buf = batch_trajectories.mutable_unchecked<3>();
        
        // 使用 Python 的 print 和 time 函数
        auto print = py::module::import("builtins").attr("print");
        auto time = py::module::import("time").attr("time");
        double start_time = time().cast<double>();
        
        for (int i = 0; i < n_particles; ++i) {
            if (i % 20 == 0 || i == n_particles - 1) {  // 每20个粒子显示一次进度
                double current_time = time().cast<double>();
                double elapsed_time = current_time - start_time;
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << elapsed_time;
                print("Computing particle", i + 1, "of", n_particles, 
                      "... (Elapsed time:", ss.str(), "seconds)");
            }
            
            // 均匀分布的位置
            double z = z_min + (z_max - z_min) * (std::rand() / (RAND_MAX + 1.0));
            // 正态分布的速度
            std::array<double, 3> v = {
                normal_random(0.0, v_std),
                normal_random(0.0, v_std),
                normal_random(0.0, v_std)
            };
            
            // Store initial state
            buf(i, 0, 0) = z;
            buf(i, 0, 1) = v[0];
            buf(i, 0, 2) = v[1];
            buf(i, 0, 3) = v[2];
            
            // Time integration
            for (int j = 1; j < n_steps; ++j) {
                double t = j * m_dt;
                std::tie(z, v) = boris_push(z, v, t, m_dt);
                
                buf(i, j, 0) = z;
                buf(i, j, 1) = v[0];
                buf(i, j, 2) = v[1];
                buf(i, j, 3) = v[2];
            }
            
            // 每处理50个粒子强制内存回收一次
            if (i % 50 == 0 && i > 0) {
                py::gil_scoped_acquire acquire;
                py::module::import("gc").attr("collect")();
            }
        }
        
        // 最终内存回收
        {
            py::gil_scoped_acquire acquire;
            py::module::import("gc").attr("collect")();
        }
        
        return batch_trajectories;
    }
};

// 确保PYBIND11_MODULE在类定义之外
PYBIND11_MODULE(solar_wind_cpp_core_v2, m) {
    py::class_<SolarWindCoreV2>(m, "SolarWindCoreV2")
        .def(py::init<const py::array_t<double>&,
                     const py::array_t<double>&,
                     const py::array_t<double>&,
                     double, double, double, double, double>())
        .def("run_particle_simulation", &SolarWindCoreV2::run_particle_simulation)
        .def("run_multi_particle_simulation", &SolarWindCoreV2::run_multi_particle_simulation,
             py::arg("n_particles"),
             py::arg("z_min"),
             py::arg("z_max"),
             py::arg("total_time"),
             py::arg("batch_size") = 500);
}