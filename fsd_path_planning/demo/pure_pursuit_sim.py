"""
Lap exploration simulation: the car starts WITHOUT knowing the cones,
discovers them with a limited range sensor, PathPlanner calculates the path
with what has been seen so far, and Pure Pursuit follows it.

Use: 
    python -m fsd_path_planning.demo.pure_pursuit_sim
    python -m fsd_path_planning.demo.pure_pursuit_sim --track fsd_path_planning/demo/fss_19_4_laps.json --gif
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.control.pure_pursuit import (
    KinematicBicycle,
    PurePursuitConfig,
    PurePursuitController,
)

DEMO_DIR = Path(__file__).parent


def load_ground_truth_map(track_file: Path):
    """Usa el último frame del JSON (mapa SLAM completo) como pista real."""
    data = json.loads(track_file.read_text())
    last = data[-1]
    cones = [np.array(c, dtype=float).reshape(-1, 2) for c in last["slam_cones"]]
    start_pos = np.array(data[0]["car_position"], dtype=float)
    start_dir = np.array(data[0]["car_direction"], dtype=float)
    return cones, start_pos, start_dir


class ConeSensor:
    """Sensor simulado: ve conos dentro de un alcance y campo de visión.
    Acumula lo visto en un 'mapa SLAM' (SLAM perfecto + ruido opcional)."""

    def __init__(self, true_cones, sensor_range=12.0, fov_deg=140.0, noise_std=0.0,
                 remove_color=False, seed=0):
        self.true_cones = true_cones
        self.range = sensor_range
        self.half_fov = np.deg2rad(fov_deg) / 2
        self.noise_std = noise_std
        self.remove_color = remove_color
        self.rng = np.random.default_rng(seed)
        # mapa: por tipo, índices ya descubiertos y su posición estimada
        self.seen = [dict() for _ in ConeTypes]

    def observe(self, position, yaw):
        for cone_type, cones in enumerate(self.true_cones):
            if len(cones) == 0:
                continue
            rel = cones - position
            dist = np.linalg.norm(rel, axis=1)
            ang = np.arctan2(rel[:, 1], rel[:, 0]) - yaw
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            visible = (dist < self.range) & (np.abs(ang) < self.half_fov)
            for i in np.flatnonzero(visible):
                if i not in self.seen[cone_type]:
                    noise = self.rng.normal(0, self.noise_std, 2) if self.noise_std > 0 else 0
                    self.seen[cone_type][i] = cones[i] + noise

    def slam_map(self):
        out = [np.array(list(d.values())).reshape(-1, 2) for d in self.seen]
        if self.remove_color:
            colorless = [np.zeros((0, 2)) for _ in ConeTypes]
            colorless[ConeTypes.UNKNOWN] = np.vstack(out)
            return colorless
        return out

    def n_seen(self):
        return sum(len(d) for d in self.seen)


def run(track_file: Path, laps: int, sensor_range: float, noise: float, remove_color: bool,
        dt: float, planner_period: float, max_time: float, cfg: PurePursuitConfig):
    true_cones, start_pos, start_dir = load_ground_truth_map(track_file)
    all_true = np.vstack([c for c in true_cones if len(c)])

    car = KinematicBicycle(*start_pos, np.arctan2(start_dir[1], start_dir[0]), v=0.0,
                           wheelbase=cfg.wheelbase)
    sensor = ConeSensor(true_cones, sensor_range, noise_std=noise, remove_color=remove_color)
    planner = PathPlanner(MissionTypes.trackdrive)
    controller = PurePursuitController(cfg)

    path = None
    log = {"xy": [], "v": [], "steer": [], "target": [], "paths": [], "seen": []}
    planner_times, failures = [], 0
    steps_per_plan = max(1, int(round(planner_period / dt)))
    distance, lap_count, left_start = 0.0, 0, False

    for step in range(int(max_time / dt)):
        sensor.observe(car.position, car.yaw)

        if step % steps_per_plan == 0:
            t0 = time.perf_counter()
            try:
                path = planner.calculate_path_in_global_frame(sensor.slam_map(), car.position, car.direction)
            except Exception as e:  # el planificador puede fallar; seguimos con la última ruta
                failures += 1
                if path is None:
                    raise RuntimeError(f"El planificador falló al inicio: {e}") from e
            planner_times.append(time.perf_counter() - t0)
            log["paths"].append(path[:, 1:3].copy())
            log["seen"].append(sensor.slam_map())

        out = controller.compute(path, car.position, car.yaw, car.v, dt)
        prev = car.position
        car.step(out.steer, out.accel, dt)
        distance += np.linalg.norm(car.position - prev)

        log["xy"].append(car.position)
        log["v"].append(car.v)
        log["steer"].append(out.steer)
        log["target"].append(out.target_point)

        # conteo de vueltas: salir de la zona de salida y regresar
        d_start = np.linalg.norm(car.position - start_pos)
        if d_start > 15:
            left_start = True
        if left_start and d_start < 3 and distance > 50:
            lap_count += 1
            left_start = False
            print(f"Vuelta {lap_count} completada: {step * dt:.1f} s, {distance:.0f} m")
            if lap_count >= laps:
                break

    xy = np.array(log["xy"])
    # holgura: distancia del eje trasero al cono más cercano en cada paso
    clearance = np.min(np.linalg.norm(xy[:, None, :] - all_true[None], axis=2), axis=1)
    results = {
        "laps": lap_count,
        "time_s": len(xy) * dt,
        "distance_m": distance,
        "v_mean": float(np.mean(log["v"])),
        "v_max": float(np.max(log["v"])),
        "min_clearance_m": float(clearance.min()),
        "cone_hits": int(np.sum(np.diff((clearance < 0.6).astype(int)) == 1)),
        "planner_ms_mean": 1000 * float(np.mean(planner_times[1:] or planner_times)),
        "planner_failures": failures,
        "cones_seen": sensor.n_seen(),
        "cones_total": len(all_true),
    }
    return results, log, true_cones, clearance


def plot(log, true_cones, clearance, out_png: Path):
    import matplotlib.pyplot as plt

    xy, v = np.array(log["xy"]), np.array(log["v"])
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {ConeTypes.UNKNOWN: "gray", ConeTypes.YELLOW: "gold", ConeTypes.BLUE: "royalblue",
              ConeTypes.ORANGE_SMALL: "orange", ConeTypes.ORANGE_BIG: "darkorange"}
    for t, cones in enumerate(true_cones):
        if len(cones):
            ax.scatter(*cones.T, c=colors[ConeTypes(t)], s=18, edgecolors="k", linewidths=0.3, zorder=3)
    sc = ax.scatter(*xy.T, c=v, cmap="viridis", s=3, zorder=2)
    fig.colorbar(sc, ax=ax, label="velocidad (m/s)")
    tight = clearance < 0.6
    if tight.any():
        ax.scatter(*xy[tight].T, c="red", s=10, zorder=4, label="holgura < 0.6 m")
        ax.legend()
    ax.set_aspect("equal")
    ax.set_title("Vuelta de exploración: PathPlanner + Pure Pursuit")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"Gráfica guardada en {out_png}")


def animate(log, true_cones, out_gif: Path, every: int = 2):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    xy = np.array(log["xy"])
    n_plans = len(log["paths"])
    steps_per_plan = len(xy) / n_plans
    all_true = np.vstack([c for c in true_cones if len(c)])
    fig, ax = plt.subplots(figsize=(8, 7))

    def draw(k):
        ax.clear()
        ax.scatter(*all_true.T, c="lightgray", s=10)  # conos aún desconocidos
        seen = log["seen"][k]
        for t, c in [(ConeTypes.BLUE, "royalblue"), (ConeTypes.YELLOW, "gold"),
                     (ConeTypes.UNKNOWN, "gray"), (ConeTypes.ORANGE_BIG, "darkorange")]:
            if len(seen[t]):
                ax.scatter(*seen[t].T, c=c, s=18, edgecolors="k", linewidths=0.3)
        i = min(int(k * steps_per_plan), len(xy) - 1)
        ax.plot(*xy[: i + 1].T, "k-", lw=1)
        ax.plot(*log["paths"][k].T, "r-", lw=2)
        ax.scatter(*log["target"][i], c="m", s=40, marker="x")
        ax.scatter(*xy[i], c="k", s=40)
        ax.set_xlim(xy[i, 0] - 25, xy[i, 0] + 25)
        ax.set_ylim(xy[i, 1] - 25, xy[i, 1] + 25)
        ax.set_aspect("equal")
        ax.set_title(f"t = {i * 0.05:.1f} s   v = {log['v'][i]:.1f} m/s")

    anim = FuncAnimation(fig, draw, frames=range(0, n_plans, every))
    anim.save(out_gif, writer=PillowWriter(fps=10))
    print(f"Animación guardada en {out_gif}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", type=Path, default=DEMO_DIR / "fsg_19_2_laps.json")
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--range", type=float, default=12.0, help="alcance del sensor (m)")
    p.add_argument("--noise", type=float, default=0.0, help="ruido de posición de conos (m)")
    p.add_argument("--no-color", action="store_true", help="probar el modo sin color")
    p.add_argument("--v-max", type=float, default=8.0)
    p.add_argument("--lookahead-gain", type=float, default=0.35)
    p.add_argument("--lookahead-min", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.05, help="periodo de control (s)")
    p.add_argument("--planner-period", type=float, default=0.1, help="periodo del planificador (s)")
    p.add_argument("--max-time", type=float, default=150.0)
    p.add_argument("--out", type=Path, default=Path("pure_pursuit_result.png"))
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()

    cfg = PurePursuitConfig(v_max=a.v_max, lookahead_gain=a.lookahead_gain, lookahead_min=a.lookahead_min)
    results, log, true_cones, clearance = run(a.track, a.laps, a.range, a.noise, a.no_color,
                                              a.dt, a.planner_period, a.max_time, cfg)
    print("\nResultados:")
    for k, v in results.items():
        print(f"  {k:18s} {v:.2f}" if isinstance(v, float) else f"  {k:18s} {v}")
    plot(log, true_cones, clearance, a.out)
    if a.gif:
        animate(log, true_cones, a.out.with_suffix(".gif"))


if __name__ == "__main__":
    main()
