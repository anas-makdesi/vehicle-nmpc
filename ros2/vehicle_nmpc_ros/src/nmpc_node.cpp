// NMPC controller node: one real-time iteration of the generated acados solver per state
// message. No Python in the loop.
//
// Subscribes  /vehicle/state   vehicle_nmpc_msgs/FrenetState
// Publishes   /vehicle/cmd     ackermann_msgs/AckermannDriveStamped
//                              (steering_angle_velocity = delta_dot, acceleration = ax,
//                               steering_angle / speed = predicted values at the next node)
//             /nmpc/stats      vehicle_nmpc_msgs/ControllerStats
// Parameters  mu_model, mu_limit (friction for the model and the constraints; can be
//             changed at run time, e.g. by a grip estimator)

#include <chrono>
#include <memory>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vehicle_nmpc_msgs/msg/controller_stats.hpp"
#include "vehicle_nmpc_msgs/msg/frenet_state.hpp"
#include "vehicle_nmpc_ros/nmpc_solver.hpp"

using vehicle_nmpc_msgs::msg::ControllerStats;
using vehicle_nmpc_msgs::msg::FrenetState;
using ackermann_msgs::msg::AckermannDriveStamped;

class NmpcNode : public rclcpp::Node
{
public:
  NmpcNode()
  : Node("nmpc")
  {
    const double mu_model = declare_parameter("mu_model", 1.0);
    const double mu_limit = declare_parameter("mu_limit", mu_model);
    solver_.set_friction(mu_model, mu_limit);
    param_cb_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & ps) {
        double mm = get_parameter("mu_model").as_double();
        double ml = get_parameter("mu_limit").as_double();
        for (const auto & p : ps) {
          if (p.get_name() == "mu_model") {mm = p.as_double();}
          if (p.get_name() == "mu_limit") {ml = p.as_double();}
        }
        rcl_interfaces::msg::SetParametersResult res;
        res.successful = mm > 0.0 && ml > 0.0;
        if (res.successful) {
          solver_.set_friction(mm, ml);
          RCLCPP_INFO(get_logger(), "friction set: model %.3f, limit %.3f", mm, ml);
        }
        return res;
      });
    auto qos = rclcpp::QoS(1).best_effort();
    cmd_pub_ = create_publisher<AckermannDriveStamped>("/vehicle/cmd", qos);
    stats_pub_ = create_publisher<ControllerStats>("/nmpc/stats", 10);
    state_sub_ = create_subscription<FrenetState>(
      "/vehicle/state", qos, [this](FrenetState::ConstSharedPtr m) {on_state(*m);});
    RCLCPP_INFO(get_logger(), "NMPC ready: N = %d, mu_model %.2f, mu_limit %.2f",
      solver_.horizon(), mu_model, mu_limit);
  }

private:
  void on_state(const FrenetState & m)
  {
    const auto t0 = std::chrono::steady_clock::now();
    const vehicle_nmpc::State x{m.s, m.n, m.xi, m.vx, m.vy, m.r, m.delta};
    const auto res = solver_.step(x);

    AckermannDriveStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = "base_link";
    cmd.drive.steering_angle_velocity = static_cast<float>(res.u[0]);
    cmd.drive.acceleration = static_cast<float>(res.u[1]);
    cmd.drive.steering_angle = static_cast<float>(res.next_delta);
    cmd.drive.speed = static_cast<float>(res.next_vx);
    cmd_pub_->publish(cmd);

    ControllerStats st;
    st.header = m.header;
    st.solve_time_ms = res.solve_time_ms;
    st.step_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
    st.latency_ms = (rclcpp::Time(cmd.header.stamp) - rclcpp::Time(m.header.stamp)).seconds() *
      1e3;
    st.status = res.status;
    st.mu_planned = solver_.mu_limit();
    stats_pub_->publish(st);
    if (res.status != 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "acados status %d", res.status);
    }
  }

  vehicle_nmpc::NmpcSolver solver_;
  rclcpp::Publisher<AckermannDriveStamped>::SharedPtr cmd_pub_;
  rclcpp::Publisher<ControllerStats>::SharedPtr stats_pub_;
  rclcpp::Subscription<FrenetState>::SharedPtr state_sub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NmpcNode>());
  rclcpp::shutdown();
  return 0;
}
