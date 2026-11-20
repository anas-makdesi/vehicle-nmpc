// Plant model of the simulated car: C++ mirror of vnmpc.model.plant_rhs / plant_step.
// Checked against the Python implementation in test/test_plant.cpp.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vehicle_nmpc
{

constexpr int NX = 7;  // s, n, xi, vx, vy, r, delta
constexpr int NU = 2;  // delta_dot, ax
using State = std::array<double, NX>;
using Input = std::array<double, NU>;

struct VehicleParams
{
  double m = 1500.0, Iz = 2250.0, lf = 1.2, lr = 1.4;
  double B = 10.0, C = 1.9, D = 1.0;  // magic formula
  double g = 9.81;
  double brake_front = 0.6;
  double L() const {return lf + lr;}
  double Fzf() const {return m * g * lr / L();}
  double Fzr() const {return m * g * lf / L();}
};

// Closed track: curvature table over arc length, linear interpolation, periodic.
class Track
{
public:
  Track() = default;
  explicit Track(const std::string & csv_path) {load(csv_path);}

  void load(const std::string & csv_path)
  {
    std::ifstream f(csv_path);
    if (!f) {throw std::runtime_error("cannot open track file " + csv_path);}
    std::string line;
    s_.clear();
    kappa_.clear();
    while (std::getline(f, line)) {
      if (line.empty()) {continue;}
      if (line[0] == '#') {  // "# length=...,half_width=..."
        auto p = line.find("length=");
        if (p != std::string::npos) {length_ = std::stod(line.substr(p + 7));}
        auto q = line.find("half_width=");
        if (q != std::string::npos) {half_width_ = std::stod(line.substr(q + 11));}
        continue;
      }
      std::stringstream ss(line);
      std::string a, b;
      std::getline(ss, a, ',');
      std::getline(ss, b, ',');
      s_.push_back(std::stod(a));
      kappa_.push_back(std::stod(b));
    }
    if (s_.size() < 2 || length_ <= 0) {throw std::runtime_error("bad track file " + csv_path);}
  }

  double kappa_at(double s) const
  {
    double sw = std::fmod(s, length_);
    if (sw < 0) {sw += length_;}
    auto it = std::upper_bound(s_.begin(), s_.end(), sw);
    std::size_t i1 = static_cast<std::size_t>(it - s_.begin());
    std::size_t i0 = i1 - 1;
    double s0 = s_[i0], k0 = kappa_[i0], s1, k1;
    if (i1 == s_.size()) {s1 = length_; k1 = kappa_[0];} else {s1 = s_[i1]; k1 = kappa_[i1];}
    double w = (sw - s0) / (s1 - s0);
    return (1 - w) * k0 + w * k1;
  }

  double length() const {return length_;}
  double half_width() const {return half_width_;}

private:
  std::vector<double> s_, kappa_;
  double length_ = 0.0, half_width_ = 2.5;
};

inline State plant_rhs(const State & x, const Input & u, const VehicleParams & p, double mu,
  double kappa)
{
  const double n = x[1], xi = x[2], vx = x[3], vy = x[4], r = x[5], delta = x[6];
  const double af = delta - std::atan2(vy + p.lf * r, vx);
  const double ar = -std::atan2(vy - p.lr * r, vx);
  const double Fx = p.m * u[1];
  double Fxf = Fx >= 0 ? 0.0 : p.brake_front * Fx;
  double Fxr = Fx >= 0 ? Fx : (1 - p.brake_front) * Fx;
  const double capf = mu * p.Fzf(), capr = mu * p.Fzr();
  Fxf = std::clamp(Fxf, -capf, capf);
  Fxr = std::clamp(Fxr, -capr, capr);
  double Fyf = mu * p.Fzf() * p.D * std::sin(p.C * std::atan(p.B * af));
  double Fyr = mu * p.Fzr() * p.D * std::sin(p.C * std::atan(p.B * ar));
  Fyf *= std::sqrt(std::max(0.0, 1.0 - (Fxf / capf) * (Fxf / capf)));
  Fyr *= std::sqrt(std::max(0.0, 1.0 - (Fxr / capr) * (Fxr / capr)));
  const double cd = std::cos(delta), sd = std::sin(delta);
  const double sdot = (vx * std::cos(xi) - vy * std::sin(xi)) / (1 - n * kappa);
  return {
    sdot,
    vx * std::sin(xi) + vy * std::cos(xi),
    r - kappa * sdot,
    (Fxf * cd - Fyf * sd + Fxr) / p.m + vy * r,
    (Fxf * sd + Fyf * cd + Fyr) / p.m - vx * r,
    (p.lf * (Fxf * sd + Fyf * cd) - p.lr * Fyr) / p.Iz,
    u[0]};
}

// RK4 over dt in `substeps` steps, zero-order hold on u; curvature frozen per substep
// at the substep's start (as in the Python plant).
inline State plant_step(State x, const Input & u, double dt, const VehicleParams & p, double mu,
  const Track & track, int substeps = 10)
{
  const double h = dt / substeps;
  auto axpy = [](const State & a, double c, const State & b) {
      State o;
      for (int i = 0; i < NX; ++i) {o[i] = a[i] + c * b[i];}
      return o;
    };
  for (int k = 0; k < substeps; ++k) {
    const double kap = track.kappa_at(x[0]);
    State k1 = plant_rhs(x, u, p, mu, kap);
    State k2 = plant_rhs(axpy(x, 0.5 * h, k1), u, p, mu, kap);
    State k3 = plant_rhs(axpy(x, 0.5 * h, k2), u, p, mu, kap);
    State k4 = plant_rhs(axpy(x, h, k3), u, p, mu, kap);
    for (int i = 0; i < NX; ++i) {x[i] += h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);}
  }
  return x;
}

}  // namespace vehicle_nmpc
