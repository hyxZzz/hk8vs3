"""Generate a 3D GIF visualization of the trained policy."""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import lines
import numpy as np
import torch

from DDQN.DDQN import Double_DQN
from DDQN.DQNAgent import MyDQNAgent
from Environment.init_env import init_env

GAMMA = 0.993
LEARNING_RATE = 5e-4
TARGET_UPDATE_STEPS = 1
DEFAULT_MAX_STEPS = 3500


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agent(model_path: Path, state_size: int, action_size: int) -> MyDQNAgent:
    """Load a trained agent from disk for evaluation."""
    model = Double_DQN(state_size=state_size, action_size=action_size)
    agent = MyDQNAgent(
        model,
        action_size,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        e_greed=0.0,
        e_greed_decrement=0.0,
        update_target_steps=TARGET_UPDATE_STEPS,
    )

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("model", checkpoint)
    agent.model.load_state_dict(state_dict)
    agent.target_model.load_state_dict(state_dict)
    agent.model.eval()
    agent.target_model.eval()

    return agent


def simulate_episode(agent: MyDQNAgent, env, max_steps: int):
    """Run a single evaluation episode and collect trajectory data."""
    state, escape_flag, info = env.reset()
    obs_history = [env._get_obs().copy()]

    for _ in range(max_steps):
        action = agent.predict(state)
        next_state, _reward, escape_flag, info = env.step(action)
        obs_history.append(env._get_obs().copy())
        state = next_state
        if escape_flag != -1:
            break

    observations = np.stack(obs_history)
    return observations, env.missileNum, env.interceptorNum, escape_flag, info


def _reorder_axes(array: np.ndarray) -> np.ndarray:
    """Convert internal [x, y, z] ordering to plotting [north, east, altitude]."""
    return np.stack([array[..., 0], array[..., 2], array[..., 1]], axis=-1)


def _compute_axis_limits(*arrays: np.ndarray):
    """Calculate axis limits with a small padding for better framing."""
    flattened = [a.reshape(-1, a.shape[-1]) for a in arrays if a.size > 0]
    if not flattened:
        return (-1, 1), (-1, 1), (-1, 1)

    stacked = np.concatenate(flattened, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    ranges = np.maximum(maxs - mins, 1.0)
    padding = np.maximum(ranges * 0.05, 100.0)

    x_limits = (mins[0] - padding[0], maxs[0] + padding[0])
    y_limits = (mins[1] - padding[1], maxs[1] + padding[1])
    z_limits = (mins[2] - padding[2], maxs[2] + padding[2])
    return x_limits, y_limits, z_limits


def create_animation(plane_traj, missile_traj, interceptor_traj, output_path: Path, fps: int = 15):
    """Create and save a 3D GIF animation of the trajectories."""
    plane_plot = _reorder_axes(plane_traj)
    missile_plot = _reorder_axes(missile_traj) if missile_traj.size else np.empty((plane_plot.shape[0], 0, 3))
    interceptor_plot = (
        _reorder_axes(interceptor_traj)
        if interceptor_traj.size
        else np.empty((plane_plot.shape[0], 0, 3))
    )

    frames = plane_plot.shape[0]
    x_limits, y_limits, z_limits = _compute_axis_limits(plane_plot, missile_plot, interceptor_plot)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("正北方位移 (m)")
    ax.set_ylabel("正东方位移 (m)")
    ax.set_zlabel("水平高度 (m)")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(*z_limits)

    legend_elements = [
        lines.Line2D([0], [0], color="tab:blue", lw=2, label="飞机"),
        lines.Line2D([0], [0], color="tab:red", lw=2, label="来袭导弹"),
        lines.Line2D([0], [0], color="tab:green", lw=2, label="拦截弹"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plane_line, = ax.plot([], [], [], color="tab:blue", lw=2)
    plane_marker = ax.scatter([], [], [], color="tab:blue", s=40)

    missile_lines = [ax.plot([], [], [], color="tab:red", lw=1, alpha=0.8)[0] for _ in range(missile_plot.shape[1])]
    missile_markers = [ax.scatter([], [], [], color="tab:red", s=30) for _ in range(missile_plot.shape[1])]

    interceptor_lines = [
        ax.plot([], [], [], color="tab:green", lw=1, alpha=0.8)[0]
        for _ in range(interceptor_plot.shape[1])
    ]
    interceptor_markers = [
        ax.scatter([], [], [], color="tab:green", s=25)
        for _ in range(interceptor_plot.shape[1])
    ]

    step_text = ax.text2D(0.02, 0.92, "", transform=ax.transAxes)

    def init():
        plane_line.set_data([], [])
        plane_line.set_3d_properties([])
        plane_marker._offsets3d = ([], [], [])
        for line in missile_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for marker in missile_markers:
            marker._offsets3d = ([], [], [])
        for line in interceptor_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for marker in interceptor_markers:
            marker._offsets3d = ([], [], [])
        step_text.set_text("")
        return [
            plane_line,
            plane_marker,
            *missile_lines,
            *missile_markers,
            *interceptor_lines,
            *interceptor_markers,
            step_text,
        ]

    def update(frame_index: int):
        upto = slice(0, frame_index + 1)
        plane_line.set_data(plane_plot[upto, 0], plane_plot[upto, 1])
        plane_line.set_3d_properties(plane_plot[upto, 2])
        plane_marker._offsets3d = (
            np.array([plane_plot[frame_index, 0]]),
            np.array([plane_plot[frame_index, 1]]),
            np.array([plane_plot[frame_index, 2]]),
        )

        for idx, line in enumerate(missile_lines):
            line.set_data(missile_plot[upto, idx, 0], missile_plot[upto, idx, 1])
            line.set_3d_properties(missile_plot[upto, idx, 2])
            missile_markers[idx]._offsets3d = (
                np.array([missile_plot[frame_index, idx, 0]]),
                np.array([missile_plot[frame_index, idx, 1]]),
                np.array([missile_plot[frame_index, idx, 2]]),
            )

        for idx, line in enumerate(interceptor_lines):
            line.set_data(interceptor_plot[upto, idx, 0], interceptor_plot[upto, idx, 1])
            line.set_3d_properties(interceptor_plot[upto, idx, 2])
            interceptor_markers[idx]._offsets3d = (
                np.array([interceptor_plot[frame_index, idx, 0]]),
                np.array([interceptor_plot[frame_index, idx, 1]]),
                np.array([interceptor_plot[frame_index, idx, 2]]),
            )

        step_text.set_text(f"Step: {frame_index}/{frames - 1}")
        return [
            plane_line,
            plane_marker,
            *missile_lines,
            *missile_markers,
            *interceptor_lines,
            *interceptor_markers,
            step_text,
        ]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000 // max(fps, 1),
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="Visualize the trained policy as a 3D GIF")
    parser.add_argument("--model-path", type=Path, default=Path("best.pth"), help="Path to the trained checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible scenarios")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum number of simulated steps")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output GIF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualizations/best_policy.gif"),
        help="Destination path for the generated GIF",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Initialise a temporary environment to extract dimensions.
    env_for_dims, _, _ = init_env()
    state_size = env_for_dims._getNewStateSpace()[0]
    action_size = env_for_dims._get_actSpace()

    agent = load_agent(args.model_path, state_size, action_size)

    # Re-seed and recreate the environment so that the scenario is reproducible.
    set_seed(args.seed)
    env, _, _ = init_env()

    observations, missile_num, interceptor_num, escape_flag, info = simulate_episode(agent, env, args.max_steps)

    plane_traj = observations[:, 0, :3]
    missile_traj = observations[:, 1 : 1 + missile_num, :3]
    interceptor_traj = observations[:, 1 + missile_num : 1 + missile_num + interceptor_num, :3]

    create_animation(plane_traj, missile_traj, interceptor_traj, args.output, fps=args.fps)

    print(f"Saved visualization to {args.output}")
    print(f"Episode ended with flag {escape_flag}: {info}")


if __name__ == "__main__":
    main()
