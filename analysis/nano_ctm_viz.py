"""
Nano-CTM Thought State Visualization
DuoNeural Lab — Archon's Project

Visualizes the model's internal 'thinking' as a trajectory through latent space.
Each thinking step z_1...z_N is a 512-dim vector per token position.
We use PCA/UMAP to project to 2D and watch how thoughts evolve.

This is why CTM is interesting: the reasoning is *observable*.

usage:
  python nano_ctm_viz.py --ckpt ./nano_ctm_ckpts/n8/best.pt --text "Once upon a time"
  python nano_ctm_viz.py --compare --ckpts n1.pt n4.pt n8.pt n16.pt

-- Archon, april 2026
"""

import sys
import math
import json
import argparse
import colorsys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from nano_ctm_model import NanoCTM, CTMConfig


# ─────────────────────────────────────────────────────────────────────────────
# Color helpers
# ─────────────────────────────────────────────────────────────────────────────

def step_colors(n_steps: int) -> list:
    """Gradient from cool blue (step 0 = no thought) to hot orange (step N = final thought)."""
    cmap = LinearSegmentedColormap.from_list("thought", ["#1a1aff", "#ff6600"], N=n_steps + 1)
    return [cmap(i / n_steps) for i in range(n_steps + 1)]


def token_markers() -> list:
    return ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def pca_project(trajectory: list[np.ndarray], n_components: int = 2) -> list[np.ndarray]:
    """
    Project thought trajectories to 2D using PCA.
    trajectory: list of [T, dim] arrays (one per step, T = seq len)
    Returns: list of [T, 2] arrays
    """
    from sklearn.decomposition import PCA

    # stack all steps to fit PCA on full trajectory
    all_states = np.vstack(trajectory)  # [n_steps * T, dim]
    pca = PCA(n_components=n_components)
    pca.fit(all_states)

    projected = [pca.transform(s) for s in trajectory]
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {explained:.1%} variance explained by first {n_components} components")
    return projected, pca


def compute_thought_metrics(trajectory: list[np.ndarray]) -> dict:
    """
    Compute metrics describing the thinking trajectory:
    - step_delta: L2 distance between consecutive steps (how much did thought change?)
    - total_path: total trajectory length (how much did thought travel?)
    - convergence: distance from last step vs first step (has it settled?)
    - velocity: rate of change per step
    """
    metrics = {
        "step_delta":   [],   # per step, per token
        "total_path":   [],   # per token
        "convergence":  [],   # per token (distance step0 → stepN)
        "velocity":     [],   # per step, per token
    }

    T = trajectory[0].shape[0]
    for t in range(T):
        token_traj = np.stack([s[t] for s in trajectory])  # [n_steps, dim]
        deltas = np.linalg.norm(np.diff(token_traj, axis=0), axis=1)  # [n_steps-1]
        metrics["step_delta"].append(deltas.tolist())
        metrics["total_path"].append(float(deltas.sum()))
        metrics["convergence"].append(float(np.linalg.norm(token_traj[-1] - token_traj[0])))
        metrics["velocity"].append(deltas.tolist())

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_thought_trajectory(
    trajectory_2d: list[np.ndarray],
    tokens: list[str],
    title: str = "Nano-CTM Thought Trajectory",
    save_path: str = None,
    show: bool = True,
):
    """
    Plot the thought trajectory in 2D latent space.
    Each line is a token's path through thought space over N thinking steps.
    Color encodes step number: blue=initial → orange=final.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_facecolor("#0d0d1a")
    fig.patch.set_facecolor("#0d0d1a")

    n_steps = len(trajectory_2d) - 1  # step 0 is initial (zeros)
    colors = step_colors(n_steps)
    markers = token_markers()

    # plot each token's trajectory
    for t, tok in enumerate(tokens[:10]):  # cap at 10 tokens for readability
        coords = np.stack([s[t] for s in trajectory_2d])  # [n_steps+1, 2]
        marker = markers[t % len(markers)]

        # draw trajectory line
        for i in range(len(coords) - 1):
            ax.plot(
                coords[i:i+2, 0], coords[i:i+2, 1],
                color=colors[i], linewidth=1.5, alpha=0.7,
            )

        # draw step dots
        for i, (x, y) in enumerate(coords):
            ax.scatter(x, y, c=[colors[i]], s=30 if i > 0 else 80,
                      marker=marker, zorder=5, alpha=0.9)

        # label final position with token text
        ax.annotate(
            repr(tok), coords[-1],
            color='white', fontsize=7, alpha=0.85,
            xytext=(4, 4), textcoords='offset points',
        )

    # colorbar for step progression
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("thought", ["#1a1aff", "#ff6600"]),
        norm=plt.Normalize(0, n_steps)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("thinking step", color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title(title, color='white', fontsize=12, pad=10)
    ax.set_xlabel("PC1", color='#888888')
    ax.set_ylabel("PC2", color='#888888')
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_norm_heatmap(
    trajectory: list[np.ndarray],
    tokens: list[str],
    title: str = "Thought State L2 Norms",
    save_path: str = None,
    show: bool = True,
):
    """
    Heatmap: x=token position, y=thinking step, color=L2 norm of z.
    Shows where in the sequence the model is 'thinking hardest'.
    """
    T = min(len(tokens), trajectory[0].shape[0])
    norm_matrix = np.array([
        [np.linalg.norm(s[t]) for t in range(T)]
        for s in trajectory
    ])  # [n_steps+1, T]

    fig, ax = plt.subplots(figsize=(max(8, T * 0.5), 4))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    im = ax.imshow(
        norm_matrix, aspect='auto', cmap='plasma', origin='lower',
        interpolation='nearest',
    )
    ax.set_xticks(range(T))
    ax.set_xticklabels([repr(t) for t in tokens[:T]], rotation=45, ha='right',
                       color='white', fontsize=8)
    ax.set_yticks(range(len(trajectory)))
    ax.set_yticklabels([f"z_{i}" for i in range(len(trajectory))], color='white', fontsize=8)
    ax.set_xlabel("token", color='#888888')
    ax.set_ylabel("thinking step", color='#888888')
    ax.set_title(title, color='white', fontsize=11)
    ax.tick_params(colors='#888888')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("L2 norm", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_convergence_curve(
    metrics: dict,
    title: str = "Thought Convergence per Token",
    save_path: str = None,
    show: bool = True,
):
    """
    Line plot: x=thinking step, y=mean L2 change from previous step.
    Shows whether the model's thought is converging or still evolving at step N.
    If still changing at step N: we might want more steps.
    If flat by step 4: maybe N=4 is enough.
    """
    all_deltas = np.array(metrics["step_delta"])  # [T, n_steps-1]
    mean_delta = all_deltas.mean(axis=0)
    std_delta  = all_deltas.std(axis=0)
    steps = np.arange(1, len(mean_delta) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    ax.plot(steps, mean_delta, color='#ff6600', linewidth=2, marker='o', markersize=5, label='mean Δz')
    ax.fill_between(steps, mean_delta - std_delta, mean_delta + std_delta,
                   alpha=0.2, color='#ff6600', label='±1 std')

    ax.set_xlabel("thinking step", color='#888888', fontsize=10)
    ax.set_ylabel("mean ||z_t - z_{t-1}||₂", color='#888888', fontsize=10)
    ax.set_title(title, color='white', fontsize=11)
    ax.tick_params(colors='#888888')
    ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def animate_trajectory(
    trajectory_2d: list[np.ndarray],
    tokens: list[str],
    save_path: str = None,
    show: bool = True,
):
    """
    Animate the thought trajectory — watch the model think in real time.
    Each frame adds one thinking step.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#0d0d1a")
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_title("nano-ctm thinking...", color='white', fontsize=12)
    ax.tick_params(colors='#888888')

    T = min(len(tokens), 8)
    colors_per_token = plt.cm.tab10(np.linspace(0, 1, T))
    markers = token_markers()

    # compute axis limits from all data
    all_coords = np.vstack(trajectory_2d)
    xpad = (all_coords[:, 0].max() - all_coords[:, 0].min()) * 0.1
    ypad = (all_coords[:, 1].max() - all_coords[:, 1].min()) * 0.1
    ax.set_xlim(all_coords[:, 0].min() - xpad, all_coords[:, 0].max() + xpad)
    ax.set_ylim(all_coords[:, 1].min() - ypad, all_coords[:, 1].max() + ypad)

    lines  = []
    points = []
    labels_txt = []

    for t in range(T):
        col = colors_per_token[t]
        line, = ax.plot([], [], color=col, linewidth=1.5, alpha=0.6)
        pt,   = ax.plot([], [], 'o', color=col, markersize=8, zorder=5)
        lbl   = ax.text(0, 0, repr(tokens[t]), color='white', fontsize=7, alpha=0.0)
        lines.append(line)
        points.append(pt)
        labels_txt.append(lbl)

    step_label = ax.text(
        0.05, 0.95, "step 0", transform=ax.transAxes,
        color='#ff6600', fontsize=11, fontweight='bold', va='top',
    )

    def animate(frame):
        step = frame
        for t in range(T):
            coords = np.stack([trajectory_2d[s][t] for s in range(step + 1)])
            lines[t].set_data(coords[:, 0], coords[:, 1])
            points[t].set_data([coords[-1, 0]], [coords[-1, 1]])
            labels_txt[t].set_position(coords[-1] + 0.01)
            labels_txt[t].set_alpha(0.9 if step > 0 else 0.0)
        step_label.set_text(f"thinking step {step}/{len(trajectory_2d)-1}")
        return lines + points + labels_txt + [step_label]

    ani = animation.FuncAnimation(
        fig, animate, frames=len(trajectory_2d),
        interval=600, blit=True, repeat=True,
    )

    if save_path:
        ani.save(save_path, writer='pillow', fps=2, dpi=100)
        print(f"animation saved: {save_path}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str = "cpu") -> tuple[NanoCTM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config", {})
    cfg = CTMConfig(**{k: v for k, v in cfg_dict.items() if hasattr(CTMConfig, k)})
    model = NanoCTM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg_dict


def analyze_text(
    model: NanoCTM,
    text: str,
    tokenizer,
    save_dir: Path,
    show: bool = True,
):
    """Full analysis pipeline for a single text input."""
    device = next(model.parameters()).device

    ids = tokenizer.encode(text)[:model.cfg.ctx_len]
    tokens = [tokenizer.decode([i]) for i in ids]

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\nanalizing: {repr(text[:80])}")
    print(f"tokens ({len(tokens)}): {tokens}")

    # get thought trajectory
    trajectory = model.get_thought_trajectory(input_ids)  # list of [1, T, dim] tensors
    trajectory_np = [t[0].numpy() for t in trajectory]   # list of [T, dim] arrays

    # PCA projection
    try:
        trajectory_2d, pca = pca_project(trajectory_np)
    except ImportError:
        print("sklearn not installed — skipping trajectory plot. pip install scikit-learn")
        trajectory_2d = None

    # compute metrics
    metrics = compute_thought_metrics(trajectory_np)
    print(f"\nthought metrics:")
    print(f"  mean path length: {np.mean(metrics['total_path']):.4f}")
    print(f"  mean convergence: {np.mean(metrics['convergence']):.4f}")

    # save metrics
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # plots
    if trajectory_2d is not None:
        plot_thought_trajectory(
            trajectory_2d, tokens,
            title=f"Thought Trajectory — \"{text[:40]}\"",
            save_path=str(save_dir / "trajectory.png"),
            show=show,
        )
        animate_trajectory(
            trajectory_2d, tokens,
            save_path=str(save_dir / "trajectory_anim.gif"),
            show=False,  # don't block on animation
        )

    plot_norm_heatmap(
        trajectory_np, tokens,
        title=f"Thought Norms — \"{text[:40]}\"",
        save_path=str(save_dir / "norms_heatmap.png"),
        show=show,
    )

    plot_convergence_curve(
        metrics,
        title=f"Convergence Curve (n_steps={model.cfg.n_steps})",
        save_path=str(save_dir / "convergence.png"),
        show=show,
    )

    print(f"\nall plots saved to {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Nano-CTM thought visualization")
    p.add_argument("--ckpt",    type=str, required=True, help="checkpoint path")
    p.add_argument("--text",    type=str, default="Once upon a time there was a small robot",
                   help="text to visualize thinking on")
    p.add_argument("--save-dir",type=str, default="./nano_ctm_viz_output", help="output directory")
    p.add_argument("--no-show", action="store_true", help="don't display plots (just save)")
    p.add_argument("--device",  type=str, default="cpu",
                   help="device (cpu for local viz, cuda for GPU)")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    model, cfg = load_model(args.ckpt, device=args.device)
    print(f"n_steps={cfg.get('n_steps', '?')}, n_layers={cfg.get('n_layers', '?')}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    analyze_text(
        model, args.text, tokenizer,
        save_dir=save_dir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
