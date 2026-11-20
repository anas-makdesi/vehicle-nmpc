"""Vehicle and tyre parameters (a generic mid-size passenger car)."""
from __future__ import annotations

from dataclasses import dataclass, field

G = 9.81


@dataclass(frozen=True)
class Tyre:
    """Simplified Pacejka "magic formula": Fy / Fz = mu * D * sin(C * atan(B * alpha))."""

    B: float = 10.0
    C: float = 1.9
    D: float = 1.0

    @property
    def peak_slip(self) -> float:
        """Slip angle (rad) at which the lateral force peaks."""
        import math

        return math.tan(math.pi / (2 * self.C)) / self.B

    @property
    def max_slope(self) -> float:
        """Cornering stiffness per unit load and unit friction, d(Fy/Fz)/d(alpha) at 0."""
        return self.B * self.C * self.D


@dataclass(frozen=True)
class Vehicle:
    m: float = 1500.0  # mass (kg)
    Iz: float = 2250.0  # yaw inertia (kg m^2)
    lf: float = 1.2  # CoG to front axle (m)
    lr: float = 1.4  # CoG to rear axle (m)
    width: float = 1.8  # (m), for plotting and track margins
    delta_max: float = 0.5  # road-wheel angle limit (rad)
    delta_rate_max: float = 0.6  # steering rate limit (rad/s)
    ax_min: float = -9.0  # braking limit (m/s^2), also bounded by friction
    ax_max: float = 3.5  # traction limit (m/s^2)
    tyre: Tyre = field(default_factory=Tyre)

    @property
    def L(self) -> float:
        return self.lf + self.lr

    @property
    def Fzf(self) -> float:
        return self.m * G * self.lr / self.L

    @property
    def Fzr(self) -> float:
        return self.m * G * self.lf / self.L
