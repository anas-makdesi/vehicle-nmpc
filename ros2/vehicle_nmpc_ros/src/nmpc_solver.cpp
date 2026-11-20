#include "vehicle_nmpc_ros/nmpc_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

extern "C" {
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_vehicle_nmpc.h"
}

namespace vehicle_nmpc
{

namespace
{
inline vehicle_nmpc_solver_capsule * cap(void * p)
{
  return static_cast<vehicle_nmpc_solver_capsule *>(p);
}
}  // namespace

NmpcSolver::NmpcSolver()
{
  auto * c = vehicle_nmpc_acados_create_capsule();
  int st = vehicle_nmpc_acados_create(c);
  if (st != 0) {
    vehicle_nmpc_acados_free_capsule(c);
    throw std::runtime_error("acados create failed with status " + std::to_string(st));
  }
  capsule_ = c;
  set_friction(1.0, 1.0);
}

NmpcSolver::~NmpcSolver()
{
  if (capsule_) {
    vehicle_nmpc_acados_free(cap(capsule_));
    vehicle_nmpc_acados_free_capsule(cap(capsule_));
  }
}

int NmpcSolver::horizon() const {return VEHICLE_NMPC_N;}

std::vector<double> NmpcSolver::time_steps() const
{
  auto * c = cap(capsule_);
  std::vector<double> ts(VEHICLE_NMPC_N);
  for (int k = 0; k < VEHICLE_NMPC_N; ++k) {
    ts[k] = c->nlp_in->Ts[k];
  }
  return ts;
}

void NmpcSolver::set_friction(double mu_model, double mu_limit)
{
  mu_model_ = mu_model;
  mu_limit_ = mu_limit;
  double p[VEHICLE_NMPC_NP] = {mu_model, mu_limit, std::sqrt(mu_limit)};
  for (int k = 0; k <= VEHICLE_NMPC_N; ++k) {
    vehicle_nmpc_acados_update_params(cap(capsule_), k, p, VEHICLE_NMPC_NP);
  }
}

void NmpcSolver::set_x0(const State & x0)
{
  auto * c = cap(capsule_);
  double x[NX];
  for (int i = 0; i < NX; ++i) {x[i] = x0[i];}
  ocp_nlp_constraints_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, c->nlp_out, 0, "lbx", x);
  ocp_nlp_constraints_model_set(c->nlp_config, c->nlp_dims, c->nlp_in, c->nlp_out, 0, "ubx", x);
}

void NmpcSolver::init_guess(const State & x0)
{
  // straight-line guess: keep speed, follow the centre line (as the Python NMPC)
  auto * c = cap(capsule_);
  const auto ts = time_steps();
  double t = 0.0;
  double zero_u[NU] = {0.0, 0.0};
  for (int k = 0; k <= VEHICLE_NMPC_N; ++k) {
    double xk[NX];
    for (int i = 0; i < NX; ++i) {xk[i] = x0[i];}
    xk[0] = x0[0] + x0[3] * t;
    if (k > 0) {xk[1] = xk[2] = xk[4] = xk[5] = 0.0;}
    ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "x", xk);
    if (k < VEHICLE_NMPC_N) {
      ocp_nlp_out_set(c->nlp_config, c->nlp_dims, c->nlp_out, c->nlp_in, k, "u", zero_u);
      t += ts[k];
    }
  }
  initialized_ = true;
}

SolveResult NmpcSolver::step(const State & x0)
{
  auto * c = cap(capsule_);
  if (!initialized_) {
    init_guess(x0);
    for (int i = 0; i < 5; ++i) {  // a few iterations to start from a good point
      set_x0(x0);
      vehicle_nmpc_acados_solve(c);
    }
  }
  set_x0(x0);
  SolveResult r;
  r.status = vehicle_nmpc_acados_solve(c);
  double t = 0.0;
  ocp_nlp_get(c->nlp_solver, "time_tot", &t);
  r.solve_time_ms = 1e3 * t;
  double u[NU], x1[NX];
  ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, 0, "u", u);
  ocp_nlp_out_get(c->nlp_config, c->nlp_dims, c->nlp_out, 1, "x", x1);
  r.u = {u[0], u[1]};
  r.next_delta = x1[6];
  r.next_vx = x1[3];
  return r;
}

}  // namespace vehicle_nmpc
