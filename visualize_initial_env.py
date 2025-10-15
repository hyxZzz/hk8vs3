"""Visualize a sampled interception scenario and export the motion as a GIF."""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from Environment.init_env import init_env
from flat_models.trajectory import Aircraft, Missiles


def collect_trajectories(
    aircraft: Aircraft,
    missiles: List[Missiles],
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance the aircraft and missiles for ``steps`` simulation ticks."""
    plane_positions = np.zeros((steps + 1, 3), dtype=float)
    missile_positions = np.zeros((len(missiles), steps + 1, 3), dtype=float)

    plane_positions[0] = [aircraft.X, aircraft.Y, aircraft.Z]
    for idx, missile in enumerate(missiles):
        missile_positions[idx, 0] = [missile.X, missile.Y, missile.Z]

    impact_radius = 50.0
    active_mask = np.ones(len(missiles), dtype=bool)

    for step in range(1, steps + 1):
        target_state = np.array([aircraft.X, aircraft.Y, aircraft.Z], dtype=float)
        for idx, missile in enumerate(missiles):
            if not active_mask[idx]:
                missile_positions[idx, step] = missile_positions[idx, step - 1]
                continue

            updated_pos = np.array(
                missile.MissilePosition(
                    target_state.tolist(), aircraft.V, aircraft.Pitch, aircraft.Heading
                ),
                dtype=float,
            )
            missile_positions[idx, step] = updated_pos

            if np.linalg.norm(updated_pos - target_state) <= impact_radius:
                active_mask[idx] = False
        plane_positions[step] = aircraft.AircraftPostition()

        if not active_mask.any():
            plane_positions[step + 1 :] = plane_positions[step]
            missile_positions[:, step + 1 :, :] = missile_positions[:, step : step + 1, :]
            break

    return plane_positions, missile_positions


def compute_plot_bounds(data: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    center = (maxs + mins) / 2.0
    max_span = spans.max()
    half = max_span / 2.0
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
        center[2] - half,
        center[2] + half,
    )


def build_animation(
    plane_positions: np.ndarray,
    missile_positions: np.ndarray,
    output_path: Path,
    fps: int,
) -> None:
    frames = plane_positions.shape[0]
    num_missiles = missile_positions.shape[0]

    all_points = np.concatenate(
        [plane_positions, missile_positions.reshape(num_missiles * frames, 3)],
        axis=0,
    )

    x_min, x_max, y_min, y_max, z_min, z_max = compute_plot_bounds(all_points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Initial Engagement Dynamics")

    cmap = plt.cm.get_cmap("tab10", num_missiles)

    plane_line, = ax.plot([], [], [], lw=2.5, color="tab:blue", label="Aircraft")
    plane_marker, = ax.plot([], [], [], marker="o", markersize=6, color="tab:blue")

    missile_lines = []
    missile_markers = []
    for idx in range(num_missiles):
        color = cmap(idx)
        line, = ax.plot([], [], [], lw=1.5, color=color, label=f"Missile {idx + 1}")
        marker, = ax.plot([], [], [], marker="^", markersize=5, color=color, linestyle="None")
        missile_lines.append(line)
        missile_markers.append(marker)

    semi_major = 20000.0
    semi_minor = 18000.0
    ellipse_angles = np.linspace(0.0, 2.0 * np.pi, 200)
    ellipse_x = plane_positions[0, 0] + semi_major * np.cos(ellipse_angles)
    ellipse_z = plane_positions[0, 2] + semi_minor * np.sin(ellipse_angles)
    ellipse_y = np.full_like(ellipse_x, plane_positions[0, 1])
    ax.plot(ellipse_x, ellipse_y, ellipse_z, linestyle="--", color="grey", alpha=0.5, label="Spawn Ellipse")

    ax.legend(loc="upper right")

    def init():
        plane_line.set_data([], [])
        plane_line.set_3d_properties([])
        plane_marker.set_data([], [])
        plane_marker.set_3d_properties([])
        for line, marker in zip(missile_lines, missile_markers):
            line.set_data([], [])
            line.set_3d_properties([])
            marker.set_data([], [])
            marker.set_3d_properties([])
        return [plane_line, plane_marker, *missile_lines, *missile_markers]

    def update(frame: int):
        plane_line.set_data(plane_positions[: frame + 1, 0], plane_positions[: frame + 1, 1])
        plane_line.set_3d_properties(plane_positions[: frame + 1, 2])
        plane_marker.set_data(plane_positions[frame : frame + 1, 0], plane_positions[frame : frame + 1, 1])
        plane_marker.set_3d_properties(plane_positions[frame : frame + 1, 2])

        for idx, (line, marker) in enumerate(zip(missile_lines, missile_markers)):
            line.set_data(missile_positions[idx, : frame + 1, 0], missile_positions[idx, : frame + 1, 1])
            line.set_3d_properties(missile_positions[idx, : frame + 1, 2])
            marker.set_data(
                missile_positions[idx, frame : frame + 1, 0],
                missile_positions[idx, frame : frame + 1, 1],
            )
            marker.set_3d_properties(missile_positions[idx, frame : frame + 1, 2])

        return [plane_line, plane_marker, *missile_lines, *missile_markers]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)


def visualize_initial_environment(
    duration: float,
    fps: int,
    num_missiles: int,
    seed: Optional[int],
    output_path: Path,
) -> Path:
    if seed is not None:
        np.random.seed(seed)

    _, aircraft_agents, missiles = init_env(num_missiles=num_missiles)
    aircraft = aircraft_agents[0]
    dt = aircraft.dt
    steps = int(duration / dt)

    plane_positions, missile_positions = collect_trajectories(aircraft, missiles, steps)
    build_animation(plane_positions, missile_positions, output_path, fps)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="Simulated time span in seconds.")
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second used when encoding the GIF animation.",
    )
    parser.add_argument(
        "--num-missiles",
        type=int,
        default=3,
        help="Number of incoming missiles to sample in the environment initialization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible spawn points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("initial_environment.gif"),
        help="Path to the output GIF file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = visualize_initial_environment(
        duration=args.duration,
        fps=args.fps,
        num_missiles=args.num_missiles,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Saved animation to {output_path.resolve()}")


if __name__ == "__main__":
    main()
