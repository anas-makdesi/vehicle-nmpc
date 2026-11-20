// The C++ wrapper around the generated solver must return the inputs the Python NMPC
// returned for the same sequence of states (same code, same warm start).
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vehicle_nmpc_ros/nmpc_solver.hpp"

using namespace vehicle_nmpc;

static std::vector<std::vector<double>> read_csv(const std::string & path)
{
  std::ifstream f(path);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {continue;}
    std::vector<double> r;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {r.push_back(std::stod(tok));}
    rows.push_back(r);
  }
  return rows;
}

TEST(Solver, ReproducesPythonClosedLoop)
{
  NmpcSolver s;
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  ASSERT_EQ(rows.size(), 150u);
  double worst = 0.0;
  for (const auto & r : rows) {
    State x;
    for (int i = 0; i < NX; ++i) {x[i] = r[i];}
    auto res = s.step(x);
    EXPECT_EQ(res.status, 0);
    worst = std::max({worst, std::abs(res.u[0] - r[7]) / 0.6, std::abs(res.u[1] - r[8]) / 9.0});
  }
  EXPECT_LT(worst, 1e-6);  // relative to the input ranges
}

TEST(Solver, FitsTheRealTimeBudget)
{
  NmpcSolver s;
  auto rows = read_csv(std::string(FIXTURES) + "/solver_seq.csv");
  std::vector<double> ms;
  for (const auto & r : rows) {
    State x;
    for (int i = 0; i < NX; ++i) {x[i] = r[i];}
    auto t0 = std::chrono::steady_clock::now();
    s.step(x);
    ms.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin() + 1, ms.end());  // the first step includes the warm-up iterations
  const double median = ms[1 + (ms.size() - 1) / 2];
  EXPECT_LT(median, 20.0);  // generous: 50 Hz budget on any CI machine
}

TEST(Solver, FrictionCanChangeAtRunTime)
{
  NmpcSolver s;
  s.set_friction(0.6, 0.6);
  EXPECT_DOUBLE_EQ(s.mu_limit(), 0.6);
  State x{0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0};
  auto res = s.step(x);
  EXPECT_EQ(res.status, 0);
  EXPECT_EQ(s.horizon(), 40);
}
