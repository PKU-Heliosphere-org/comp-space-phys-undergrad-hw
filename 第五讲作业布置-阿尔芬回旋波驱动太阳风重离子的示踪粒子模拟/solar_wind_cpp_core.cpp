#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

class SolarWindCore {
public:
    SolarWindCore(const py::array_t<double>& omega,
                  const py::array_t<double>& k_z,
                  const py::array_t<double>& B_j,
                  double epsilon_0,
                  double tau,
                  double delta_tau) 
        : m_epsilon_0(epsilon_0), m_tau(tau), m_delta_tau(delta_tau) {
        
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

    double wave_amplitude_time_dependence(double t) const {
        if (t < m_tau) {
            return m_epsilon_0 * std::exp(-std::pow(t - m_tau, 2) / std::pow(m_delta_tau, 2));
        }
        return m_epsilon_0;
    }

    std::tuple<double, double, double, double> calculate_fields(double z, double t) const {
        double Bx = 0.0, By = 0.0, Ex = 0.0, Ey = 0.0;
        
        const double time_amplitude = wave_amplitude_time_dependence(t);
        
        #pragma omp parallel for reduction(+:Bx,By,Ex,Ey)
        for (int j = 0; j < m_N; ++j) {
            const double phase = m_k_z[j] * z - m_omega[j] * t;
            const double amplitude = m_B_j[j] * time_amplitude;
            
            // Left-hand circular polarization in x-y plane
            Bx += amplitude * std::cos(phase);
            By += amplitude * std::sin(phase);
            
            // Electric field components
            const double E_amplitude = amplitude * (m_omega[j] / m_k_z[j]);
            Ex += -E_amplitude * std::sin(phase);
            Ey += E_amplitude * std::cos(phase);
        }
        
        return std::make_tuple(Ex, Ey, Bx, By);
    }

    std::tuple<double, std::array<double, 3>> boris_push(
        double z, const std::array<double, 3>& v, double t, double dt) const {
        
        const double q_m_ratio = 5.0/16.0;  // Charge-to-mass ratio for O5+
        
        // Get fields at current position
        auto [Ex, Ey, Bx, By] = calculate_fields(z, t);
        const double Bz = 1.0;  // Background field
        const double Ez = 0.0;  // No parallel electric field
        
        // Construct field vectors
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
        
        double t_squared = t_vec[0]*t_vec[0] + t_vec[1]*t_vec[1] + t_vec[2]*t_vec[2];
        
        // Boris rotation factors
        std::array<double, 3> s_vec;
        for (int i = 0; i < 3; ++i) {
            s_vec[i] = 2.0 * t_vec[i] / (1.0 + t_squared);
        }
        
        // Cross product v_minus × t_vec
        std::array<double, 3> v_prime = {
            v_minus[1]*t_vec[2] - v_minus[2]*t_vec[1],
            v_minus[2]*t_vec[0] - v_minus[0]*t_vec[2],
            v_minus[0]*t_vec[1] - v_minus[1]*t_vec[0]
        };
        
        // Add to v_minus
        for (int i = 0; i < 3; ++i) {
            v_prime[i] += v_minus[i];
        }
        
        // Cross product v_prime × s_vec
        std::array<double, 3> v_plus = {
            v_minus[0] + (v_prime[1]*s_vec[2] - v_prime[2]*s_vec[1]),
            v_minus[1] + (v_prime[2]*s_vec[0] - v_prime[0]*s_vec[2]),
            v_minus[2] + (v_prime[0]*s_vec[1] - v_prime[1]*s_vec[0])
        };
        
        // 3. Half acceleration in E field
        std::array<double, 3> new_v;
        for (int i = 0; i < 3; ++i) {
            new_v[i] = v_plus[i] + 0.5 * dt * q_m_ratio * E[i];
        }
        
        // 4. Position update (using average of old and new velocities)
        double new_z = z + 0.5 * dt * (v[2] + new_v[2]);
        
        return std::make_tuple(new_z, new_v);
    }

private:
    int m_N;
    std::vector<double> m_omega;
    std::vector<double> m_k_z;
    std::vector<double> m_B_j;
    double m_epsilon_0;
    double m_tau;
    double m_delta_tau;
};

PYBIND11_MODULE(solar_wind_cpp_core, m) {
    py::class_<SolarWindCore>(m, "SolarWindCore")
        .def(py::init<const py::array_t<double>&,
                     const py::array_t<double>&,
                     const py::array_t<double>&,
                     double, double, double>())
        .def("calculate_fields", &SolarWindCore::calculate_fields)
        .def("boris_push", &SolarWindCore::boris_push);
}