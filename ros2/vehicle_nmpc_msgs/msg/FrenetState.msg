# Vehicle state in path (Frenet) coordinates; see vnmpc.model for definitions.
std_msgs/Header header
float64 s      # progress along the centre line (m)
float64 n      # lateral offset, left > 0 (m)
float64 xi     # heading relative to the centre line (rad)
float64 vx     # longitudinal speed (m/s)
float64 vy     # lateral speed (m/s)
float64 r      # yaw rate (rad/s)
float64 delta  # road-wheel steering angle (rad)
