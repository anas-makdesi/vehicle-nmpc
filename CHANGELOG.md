# Changelog

## Unreleased
- ddma pinned to `v0.12.0` (the `grip` extra). CI installs it, so the grip
  certificate tests now run on every push instead of being skipped.

## 0.1.0 (2026-11-16)
First release.
- Dynamic bicycle model in path coordinates, magic-formula tyres with per-axle friction
  circle; closed spline track with friction-circle speed profile.
- acados NMPC (exact-Hessian SQP, real-time iteration or converged), HPIPM, non-uniform
  2 s horizon; friction as a run-time parameter.
- Linear time-varying MPC baseline with OSQP (linear tyres, decoupled speed loop).
- CasADi + IPOPT reference solution of the same problem.
- Grip certificate from tyre data with ddma's `GrowthMap`; ramp-steer calibration run.
- Lap experiments and timing benchmark with figures and JSON results.
- ROS 2 Jazzy packages: messages, C++ NMPC node on the generated solver, 1 kHz C++
  simulator node, launch file, gtest parity tests, launch test, lap probe.
- CI (ruff, pytest, colcon build/test in a ROS container), Dockerfile.
