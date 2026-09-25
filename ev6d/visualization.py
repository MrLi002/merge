"""Interactive, offline 3D comparison of estimated and reference poses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .evaluation import _validate_trajectory


def visualize_tracking(dataset, result, output=None):
    """Write a self-contained HTML viewer aligned to the evaluation timestamps."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ValueError('Plotly is required: python -m pip install -e ".[visualization]"') from exc

    dataset, result = Path(dataset), Path(result)
    output = Path(output) if output is not None else result / "trajectory_3d.html"
    with np.load(dataset / "ground_truth.npz", allow_pickle=False) as archive:
        tg, pg, qg = _validate_trajectory(archive, "ground truth")
    with np.load(result / "trajectory.npz", allow_pickle=False) as archive:
        te, pe, qe = _validate_trajectory(archive, "estimate")

    valid = (te >= tg[0]) & (te <= tg[-1])
    if not np.any(valid):
        raise ValueError("No overlapping ground-truth and estimate timestamps")
    t, p, q = te[valid], pe[valid], qe[valid]
    reference_p = np.column_stack([np.interp(t, tg, pg[:, i]) for i in range(3)])
    reference_q = Slerp(tg, Rotation.from_quat(qg))(t).as_quat()
    position_error = np.linalg.norm(p - reference_p, axis=1)
    rotation_error = np.rad2deg(
        (Rotation.from_quat(q) * Rotation.from_quat(reference_q).inv()).magnitude()
    )

    metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    model_size = np.asarray(metadata.get("model", {}).get("size", []), dtype=float)
    span = np.ptp(np.vstack((p, reference_p)), axis=0)
    axis_length = 0.5 * float(np.max(model_size)) if model_size.size == 3 else 0.2 * float(np.max(span))
    axis_length = max(axis_length, 0.01)
    bounds = np.vstack((p, pg))
    lower = np.min(bounds, axis=0) - axis_length * 1.5
    upper = np.max(bounds, axis=0) + axis_length * 1.5

    estimate_color, reference_color = "#e76f51", "#277da1"
    axis_colors = ("#d62828", "#2a9d8f", "#4361ee")

    def point_trace(point, name, color, timestamp, pos_error, rot_error):
        return go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode="markers", name=name, marker=dict(size=7, color=color),
            customdata=[[timestamp, pos_error * 1000, rot_error]],
            hovertemplate=(name + "<br>t=%{customdata[0]:.3f} s"
                           + "<br>position error=%{customdata[1]:.2f} mm"
                           + "<br>rotation error=%{customdata[2]:.2f}°<extra></extra>"),
        )

    def current_traces(index):
        current = [
            point_trace(p[index], "Estimate at t", estimate_color, t[index], position_error[index], rotation_error[index]),
            point_trace(reference_p[index], "Ground truth at t", reference_color, t[index], position_error[index], rotation_error[index]),
            go.Scatter3d(
                x=[p[index, 0], reference_p[index, 0]],
                y=[p[index, 1], reference_p[index, 1]],
                z=[p[index, 2], reference_p[index, 2]],
                mode="lines", name="Position error", showlegend=False,
                line=dict(color="#6c757d", width=4), hoverinfo="skip",
            ),
        ]
        for origin, quaternion, opacity, width in (
            (p[index], q[index], 1.0, 7),
            (reference_p[index], reference_q[index], 0.45, 4),
        ):
            directions = Rotation.from_quat(quaternion).apply(np.eye(3) * axis_length)
            for direction, color in zip(directions, axis_colors):
                tip = origin + direction
                current.append(go.Scatter3d(
                    x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
                    mode="lines", showlegend=False, hoverinfo="skip", opacity=opacity,
                    line=dict(color=color, width=width),
                ))
        return current

    def title(index):
        return ("Estimated vs ground-truth 6-DoF trajectory"
                f"<br><sup>t={t[index]:.3f} s · position error={position_error[index]*1000:.2f} mm"
                f" · rotation error={rotation_error[index]:.2f}°"
                " · RGB axes: estimate bright, truth faint</sup>")

    traces = [
        go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", name="Estimate",
            line=dict(color=estimate_color, width=7),
            customdata=np.column_stack((t, position_error * 1000, rotation_error)),
            hovertemplate="Estimate<br>t=%{customdata[0]:.3f} s<br>position error=%{customdata[1]:.2f} mm"
                          "<br>rotation error=%{customdata[2]:.2f}°<extra></extra>",
        ),
        go.Scatter3d(
            x=pg[:, 0], y=pg[:, 1], z=pg[:, 2], mode="lines", name="Ground truth",
            line=dict(color=reference_color, width=6),
            customdata=tg,
            hovertemplate="Ground truth<br>t=%{customdata:.3f} s<extra></extra>",
        ),
        *current_traces(0),
    ]
    dynamic_indices = list(range(2, len(traces)))
    frames = [go.Frame(name=str(i), data=current_traces(i), traces=dynamic_indices,
                       layout=go.Layout(title=title(i))) for i in range(len(t))]
    slider_steps = [
        dict(label=f"{timestamp:.2f}", method="animate",
             args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))])
        for i, timestamp in enumerate(t)
    ]
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=title(0),
        scene=dict(
            xaxis=dict(title="Event camera X (m)", range=[lower[0], upper[0]]),
            yaxis=dict(title="Event camera Y (m)", range=[lower[1], upper[1]]),
            zaxis=dict(title="Event camera Z (m)", range=[lower[2], upper[2]]),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=75, b=0),
        legend=dict(x=0.01, y=0.99),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True),
                                      transition=dict(duration=0), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(active=0, currentvalue=dict(prefix="Time (s): "),
                      pad=dict(t=35), steps=slider_steps)],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True, full_html=True, auto_play=False)
    return {
        "output": str(output.resolve()),
        "samples": int(len(t)),
        "excluded_outside_gt_support": int((~valid).sum()),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_error**2))),
    }
