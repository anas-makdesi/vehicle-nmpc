# ROS 2 Jazzy + acados + the vehicle NMPC nodes.
#   docker build -f docker/Dockerfile -t vehicle-nmpc .
#   docker run --rm -it vehicle-nmpc                       # dry road
#   docker run --rm -it vehicle-nmpc mu:=0.6 mu_plan:=0.57 # low grip, certified friction
FROM ros:jazzy-ros-base
SHELL ["/bin/bash", "-c"]
ENV ACADOS_SOURCE_DIR=/opt/acados
RUN apt-get update -qq && apt-get install -y -qq python3-venv cmake git \
      ros-jazzy-ackermann-msgs > /dev/null && rm -rf /var/lib/apt/lists/*
# code generation runs in a venv so the ROS Python stays untouched
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install -q --upgrade pip && \
    /opt/venv/bin/pip install -q numpy scipy casadi osqp matplotlib
COPY scripts/install_acados.sh /tmp/install_acados.sh
RUN PYTHON=/opt/venv/bin/python /tmp/install_acados.sh /opt/acados
WORKDIR /opt/vehicle-nmpc
COPY . .
RUN LD_LIBRARY_PATH=/opt/acados/lib /opt/venv/bin/python scripts/export_c_code.py && \
    mkdir -p /ws && ln -s /opt/vehicle-nmpc/ros2 /ws/src && cd /ws && \
    source /opt/ros/jazzy/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
ENTRYPOINT ["/bin/bash", "-c", "source /ws/install/setup.bash && exec ros2 launch vehicle_nmpc_ros closed_loop.launch.py \"$@\"", "--"]
