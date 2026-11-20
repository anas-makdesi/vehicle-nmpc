// Vehicle simulator node: integrates the plant (vehicle_model.hpp) in real time at
// 1 kHz with the latest command, and publishes the state at the control rate.
//
// Publishes   /vehicle/state   vehicle_nmpc_msgs/FrenetState  (rate: state_rate_hz)
// Subscribes  /vehicle/cmd     ackermann_msgs/AckermannDriveStamped
// Parameters  track_csv, mu (true road friction), state_rate_hz, v0 (initial speed)
//
// Because the plant runs at 1 kHz, each command takes effect at the millisecond it
// arrives: the controller's computation and transport delay is part of the loop.

#include <chrono>
#include <memory>
#include <string>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vehicle_nmpc_msgs/msg/frenet_state.hpp"
#include "vehicle_nmpc_ros/vehicle_model.hpp"

using vehicle_nmpc_msgs::msg::FrenetState;
using ackermann_msgs::msg::AckermannDriveStamped;
using namespace std::chrono_literals;

class SimNode : public rclcpp::Node
{
public:
  SimNode()
  : Node("vehicle_sim")
  {
    const std::string csv = declare_parameter("track_csv", std::string(""));
    mu_ = declare_parameter("mu", 1.0);
    const double rate = declare_parameter("state_rate_hz", 50.0);
    const double v0 = declare_parameter("v0", 15.0);
    track_.load(csv);
    x_ = {0.0, 0.0, 0.0, v0, 0.0, 0.0, 0.0};
    publish_every_ = std::max(1, static_cast<int>(std::lround(1000.0 / rate)));
    auto qos = rclcpp::QoS(1).best_effort();
    state_pub_ = create_publisher<FrenetState>("/vehicle/state", qos);
    cmd_sub_ = create_subscription<AckermannDriveStamped>(
      "/vehicle/cmd", qos, [this](AckermannDriveStamped::ConstSharedPtr m) {
        u_ = {m->drive.steering_angle_velocity, m->drive.acceleration};
      });
    timer_ = create_wall_timer(1ms, [this]() {tick();});
    RCLCPP_INFO(get_logger(), "simulating %.0f m track, friction %.2f", track_.length(), mu_);
  }

private:
  void tick()
  {
    if (k_ % publish_every_ == 0) {
      FrenetState m;
      m.header.stamp = now();
      m.header.frame_id = "track";
      m.s = x_[0]; m.n = x_[1]; m.xi = x_[2]; m.vx = x_[3]; m.vy = x_[4]; m.r = x_[5];
      m.delta = x_[6];
      state_pub_->publish(m);
    }
    x_ = vehicle_nmpc::plant_step(x_, u_, 1e-3, p_, mu_, track_, 1);
    ++k_;
  }

  vehicle_nmpc::VehicleParams p_;
  vehicle_nmpc::Track track_;
  vehicle_nmpc::State x_{};
  vehicle_nmpc::Input u_{0.0, 0.0};
  double mu_ = 1.0;
  long k_ = 0;
  int publish_every_ = 20;
  rclcpp::Publisher<FrenetState>::SharedPtr state_pub_;
  rclcpp::Subscription<AckermannDriveStamped>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimNode>());
  rclcpp::shutdown();
  return 0;
}
