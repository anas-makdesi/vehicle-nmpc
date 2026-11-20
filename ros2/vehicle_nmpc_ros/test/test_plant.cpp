// The C++ plant must reproduce the Python plant (vnmpc.model.plant_step).
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vehicle_nmpc_ros/vehicle_model.hpp"

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

TEST(Plant, MatchesPythonStepByStep)
{
  Track track(TRACK_CSV);
  VehicleParams p;
  auto rows = read_csv(std::string(FIXTURES) + "/plant_traj.csv");
  ASSERT_EQ(rows.size(), 300u);
  double worst = 0.0;
  for (const auto & r : rows) {
    State x;
    Input u{r[7], r[8]};
    for (int i = 0; i < NX; ++i) {x[i] = r[i];}
    State xn = plant_step(x, u, 0.02, p, 0.8, track, 10);
    for (int i = 0; i < NX; ++i) {
      worst = std::max(worst, std::abs(xn[i] - r[9 + i]) / (1.0 + std::abs(r[9 + i])));
    }
  }
  EXPECT_LT(worst, 1e-9);
}

TEST(Track, PeriodicCurvature)
{
  Track track(TRACK_CSV);
  EXPECT_NEAR(track.kappa_at(10.0), track.kappa_at(10.0 + track.length()), 1e-12);
  EXPECT_NEAR(track.kappa_at(-5.0), track.kappa_at(track.length() - 5.0), 1e-12);
  EXPECT_GT(track.length(), 1000.0);
}
