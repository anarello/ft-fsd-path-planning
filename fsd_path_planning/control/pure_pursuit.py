"""
Pure Pursuit controller for following a PathPlanner trajectory.

The input trajectory is the output of
`PathPlanner.calculate_path_in_global_frame`: an Mx4 array containing
[s (m), x, y, curvature]. The vehicle's position is assumed to be at the rear axle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PurePursuitConfig:
    wheelbase: float = 1.53  # distancia entre ejes (m), típico en FS
    lookahead_gain: float = 0.35  # Ld = gain * v + Ld_min
    lookahead_min: float = 3.0  # m
    lookahead_max: float = 9.0  # m
    max_steer: float = np.deg2rad(25)  # rad
    max_steer_rate: float = np.deg2rad(180)  # rad/s, suaviza el volante
    v_max: float = 8.0  # m/s, conservador para la vuelta de exploración
    v_min: float = 2.0  # m/s
    a_lat_max: float = 6.0  # m/s^2, defines the maximum lateral acceleration
    a_long_max: float = 3.0  # m/s^2 (acceleration)
    a_brake_max: float = 6.0  # m/s^2 (braking)
    speed_kp: float = 1.5  # speed control gain


@dataclass
class ControlOutput:
    steer: float  # rad, positivo = izquierda
    accel: float  # m/s^2
    target_point: np.ndarray  # punto objetivo (x, y) en marco global
    lookahead: float  # m
    target_speed: float  # m/s


class PurePursuitController:
    def __init__(self, config: PurePursuitConfig | None = None) -> None:
        self.cfg = config or PurePursuitConfig()
        self._last_steer = 0.0

    # Geometry: lookahead and target point
    def lookahead_distance(self, speed: float) -> float:
        ld = self.cfg.lookahead_gain * speed + self.cfg.lookahead_min
        return float(np.clip(ld, self.cfg.lookahead_min, self.cfg.lookahead_max))

    @staticmethod
    def find_target_point(path_xy: np.ndarray, position: np.ndarray, lookahead: float) -> np.ndarray:
        """Primer punto de la trayectoria (hacia adelante desde el más cercano)
        a distancia >= lookahead. Interpola entre muestras para que el objetivo
        no 'salte'. Si la trayectoria es más corta, la prolonga en línea recta."""
        dists = np.linalg.norm(path_xy - position, axis=1)
        i0 = int(np.argmin(dists))
        forward = path_xy[i0:]
        d_fwd = dists[i0:]

        idx = np.flatnonzero(d_fwd >= lookahead)
        if len(idx) == 0:
            # prolongar con la dirección del último segmento
            if len(forward) >= 2:
                direction = forward[-1] - forward[-2]
                direction /= np.linalg.norm(direction) + 1e-9
            else:
                direction = forward[-1] - position
                direction /= np.linalg.norm(direction) + 1e-9
            missing = lookahead - d_fwd[-1]
            return forward[-1] + direction * max(missing, 0.0)

        j = int(idx[0])
        if j == 0:
            return forward[0]
        # interpolación lineal entre forward[j-1] y forward[j]
        d_a, d_b = d_fwd[j - 1], d_fwd[j]
        t = (lookahead - d_a) / (d_b - d_a + 1e-9)
        return forward[j - 1] + t * (forward[j] - forward[j - 1])

    # speed control: target speed based on curvature and braking distance
    def target_speed(self, path: np.ndarray, position: np.ndarray, speed: float) -> float:
        """Target speed based on curvature and braking distance."""
        path_xy, curvature = path[:, 1:3], np.abs(path[:, 3])
        dists = np.linalg.norm(path_xy - position, axis=1)
        braking_distance = speed**2 / (2 * self.cfg.a_brake_max) + 2.0
        window = dists <= max(braking_distance, self.cfg.lookahead_min)
        k_max = curvature[window].max() if window.any() else curvature.max()
        v_curve = np.sqrt(self.cfg.a_lat_max / max(k_max, 1e-3))
        return float(np.clip(v_curve, self.cfg.v_min, self.cfg.v_max))

    # control law
    def compute(self, path: np.ndarray, position: np.ndarray, yaw: float, speed: float, dt: float) -> ControlOutput:
        path_xy = path[:, 1:3]
        ld = self.lookahead_distance(speed)
        target = self.find_target_point(path_xy, position, ld)

        # objective in local frame
        dx, dy = target - position
        x_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        y_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        alpha = np.arctan2(y_local, x_local)
        ld_real = max(np.hypot(x_local, y_local), 1e-3)

        # delta = atan(2 L sin(alpha) / Ld)
        steer = np.arctan2(2.0 * self.cfg.wheelbase * np.sin(alpha), ld_real)
        steer = float(np.clip(steer, -self.cfg.max_steer, self.cfg.max_steer))

        # limit of velocity of steering wheel|
        max_delta = self.cfg.max_steer_rate * dt
        steer = float(np.clip(steer, self._last_steer - max_delta, self._last_steer + max_delta))
        self._last_steer = steer

        v_ref = self.target_speed(path, position, speed)
        accel = self.cfg.speed_kp * (v_ref - speed)
        accel = float(np.clip(accel, -self.cfg.a_brake_max, self.cfg.a_long_max))

        return ControlOutput(steer, accel, target, ld, v_ref)


class KinematicBicycle:
    """Classic kinematic bicycle model for simulating vehicle motion. The vehicle's position is assumed to be at the rear axle."""

    def __init__(self, x: float, y: float, yaw: float, v: float = 0.0, wheelbase: float = 1.53):
        self.x, self.y, self.yaw, self.v, self.L = x, y, yaw, v, wheelbase

    def step(self, steer: float, accel: float, dt: float) -> None:
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.L * np.tan(steer) * dt
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
        self.v = max(0.0, self.v + accel * dt)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])
