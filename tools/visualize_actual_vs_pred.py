#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from datasets.io import IO


def _load_points(path: str) -> np.ndarray:
    pts = IO.get(path).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point cloud at {path}, got shape {pts.shape}")
    return pts


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _make_trace(points: np.ndarray, name: str, point_size: int):
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        name=name,
        marker=dict(size=point_size, color=points[:, 2], colorscale='Viridis', opacity=0.95),
    )


def _scene_layout(points: np.ndarray):
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = (pmin + pmax) / 2.0
    span = float(np.max(pmax - pmin))
    if span <= 1e-8:
        span = 1.0
    half = span / 2.0

    return dict(
        xaxis=dict(visible=False, range=[center[0] - half, center[0] + half]),
        yaxis=dict(visible=False, range=[center[1] - half, center[1] + half]),
        zaxis=dict(visible=False, range=[center[2] - half, center[2] + half]),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    )


def main():
    parser = argparse.ArgumentParser(description='Visualize actual vs predicted point cloud in interactive 3D')
    parser.add_argument('--input_pc', required=True, help='Path to original input point cloud (.pcd/.ply/.npy/.txt/.h5)')
    parser.add_argument('--pred_pc', required=True, help='Path to predicted point cloud (typically fine.npy)')
    parser.add_argument('--output_html', default='', help='Output HTML path (default: next to pred file)')
    parser.add_argument('--title', default='AdaPoinTr: Actual vs Predicted', help='Figure title')
    parser.add_argument('--max_points', type=int, default=20000, help='Max points to render per cloud (0 disables sampling)')
    parser.add_argument('--point_size', type=int, default=2, help='Scatter point size')
    parser.add_argument('--show', action='store_true', help='Auto-open HTML in browser after creation')
    args = parser.parse_args()

    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            'plotly is required for this visualization. Install with: pip install plotly'
        ) from exc

    input_points = _sample_points(_load_points(args.input_pc), args.max_points)
    pred_points = _sample_points(_load_points(args.pred_pc), args.max_points)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Actual Input Point Cloud', 'Predicted Completed Point Cloud'),
        horizontal_spacing=0.04,
    )

    fig.add_trace(_make_trace(input_points, 'Actual', args.point_size), row=1, col=1)
    fig.add_trace(_make_trace(pred_points, 'Predicted', args.point_size), row=1, col=2)

    fig.update_layout(
        title=args.title,
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=_scene_layout(input_points),
        scene2=_scene_layout(pred_points),
    )

    output_html = args.output_html
    if not output_html:
        pred_dir = os.path.dirname(os.path.abspath(args.pred_pc))
        output_html = os.path.join(pred_dir, 'actual_vs_pred_3d.html')
    os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)

    fig.write_html(output_html, include_plotlyjs='cdn', auto_open=args.show)
    print(f'[3D View] Saved interactive comparison to: {output_html}')


if __name__ == '__main__':
    main()
