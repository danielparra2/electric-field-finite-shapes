"""Electric field of finite charge distributions by direct integration.

Units: K = 1/(4*pi*eps0) is set to 1, so charges and fields are dimensionless.

Run:
    python campo.py          # checks numerics against closed forms, writes figures/*.html
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.spatial import cKDTree

K = 1.0
FIG_DIR = Path(__file__).parent / "figures"


# --------------------------------------------------------------------------
# Quadrature: every shape becomes a cloud of point charges (positions, weights)
# --------------------------------------------------------------------------
def _gauss(n: int, a: float, b: float):
    x, w = leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def sample_wire(L: float, lam: float = 1.0, n: int = 400):
    """Straight wire along z, from -L/2 to L/2, linear density lam."""
    z, w = _gauss(n, -L / 2, L / 2)
    pts = np.column_stack([np.zeros(n), np.zeros(n), z])
    return pts, lam * w


def sample_ring(R: float, Q: float = 1.0, n: int = 720):
    """Thin ring of radius R in the plane z = 0, total charge Q."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([R * np.cos(phi), R * np.sin(phi), np.zeros(n)])
    return pts, np.full(n, Q / n)


def sample_disk(R: float, Q: float = 1.0, nr: int = 60, nphi: int = 180):
    """Uniform disk of radius R in the plane z = 0, total charge Q."""
    r, wr = _gauss(nr, 0.0, R)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    dphi = 2 * np.pi / nphi
    sigma = Q / (np.pi * R**2)
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    W = sigma * rr * wr[:, None] * dphi
    pts = np.column_stack(
        [(rr * np.cos(pp)).ravel(), (rr * np.sin(pp)).ravel(), np.zeros(rr.size)]
    )
    return pts, W.ravel()


def sample_cylinder(R: float, L: float, Q: float = 1.0,
                    nr: int = 30, nphi: int = 90, nz: int = 60):
    """Solid cylinder of radius R, length L along z, uniform volume density."""
    r, wr = _gauss(nr, 0.0, R)
    z, wz = _gauss(nz, -L / 2, L / 2)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    dphi = 2 * np.pi / nphi
    rho_v = Q / (np.pi * R**2 * L)
    RR, PP, ZZ = np.meshgrid(r, phi, z, indexing="ij")
    W = rho_v * RR * wr[:, None, None] * dphi * wz[None, None, :]
    pts = np.column_stack(
        [(RR * np.cos(PP)).ravel(), (RR * np.sin(PP)).ravel(), ZZ.ravel()]
    )
    return pts, W.ravel()


def field(query: np.ndarray, pts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """E(r) = K * sum_i w_i (r - r_i) / |r - r_i|^3, evaluated at every query point."""
    query = np.atleast_2d(query)
    E = np.zeros_like(query, dtype=float)
    chunk = max(1, int(2e6 // len(pts)))
    for i in range(0, len(query), chunk):
        d = query[i:i + chunk, None, :] - pts[None, :, :]
        r = np.linalg.norm(d, axis=2)
        E[i:i + chunk] = K * np.sum((w[None, :] / r**3)[:, :, None] * d, axis=1)
    return E


# --------------------------------------------------------------------------
# Closed forms (used as checks)
# --------------------------------------------------------------------------
def wire_closed(P: np.ndarray, L: float, lam: float = 1.0) -> np.ndarray:
    """Finite wire: E_rho = K lam/rho [a/sqrt(rho^2+a^2) - b/sqrt(rho^2+b^2)],
    E_z = K lam [1/sqrt(rho^2+b^2) - 1/sqrt(rho^2+a^2)], a = z+L/2, b = z-L/2."""
    x, y, z = P.T
    rho = np.hypot(x, y)
    a, b = z + L / 2, z - L / 2
    Erho = K * lam / rho * (a / np.sqrt(rho**2 + a**2) - b / np.sqrt(rho**2 + b**2))
    Ez = K * lam * (1 / np.sqrt(rho**2 + b**2) - 1 / np.sqrt(rho**2 + a**2))
    return np.column_stack([Erho * x / rho, Erho * y / rho, Ez])


def ring_axis_closed(z: np.ndarray, R: float, Q: float = 1.0) -> np.ndarray:
    return K * Q * z / (z**2 + R**2) ** 1.5


def disk_axis_closed(z: np.ndarray, R: float, Q: float = 1.0) -> np.ndarray:
    sigma = Q / (np.pi * R**2)
    return 2 * np.pi * K * sigma * np.sign(z) * (1 - np.abs(z) / np.sqrt(z**2 + R**2))


def cylinder_axis_closed(z: float, R: float, L: float, Q: float = 1.0) -> float:
    """Sum of thin disks of thickness dz', each with the disk on-axis field."""
    rho_v = Q / (np.pi * R**2 * L)

    def slice_field(zp):
        u = z - zp
        return 2 * np.pi * K * rho_v * np.sign(u) * (1 - abs(u) / np.sqrt(u**2 + R**2))

    val, _ = quad(slice_field, -L / 2, L / 2, points=[z] if abs(z) < L / 2 else None,
                  limit=200)
    return val


def run_checks() -> None:
    print("Numerical integration vs closed forms (relative error)")
    # Wire, off-axis point
    P = np.array([[0.7, 0.3, 0.4], [1.2, -0.5, -0.8], [0.4, 0.4, 1.9]])
    pts, w = sample_wire(L=2.0)
    err = np.linalg.norm(field(P, pts, w) - wire_closed(P, 2.0), axis=1) / \
        np.linalg.norm(wire_closed(P, 2.0), axis=1)
    print(f"  wire      max error: {err.max():.2e}")

    # Ring, on the axis
    z = np.array([0.3, 1.0, 2.5])
    pts, w = sample_ring(R=1.0)
    Enum = field(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]), pts, w)[:, 2]
    print(f"  ring      max error: {np.max(np.abs(Enum - ring_axis_closed(z, 1.0)) / np.abs(ring_axis_closed(z, 1.0))):.2e}")

    # Disk, on the axis
    pts, w = sample_disk(R=1.0)
    Enum = field(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]), pts, w)[:, 2]
    print(f"  disk      max error: {np.max(np.abs(Enum - disk_axis_closed(z, 1.0)) / np.abs(disk_axis_closed(z, 1.0))):.2e}")

    # Cylinder, on the axis
    pts, w = sample_cylinder(R=1.0, L=2.0)
    # Exterior points only: inside the volume the 1/r^2 singularity of the
    # integrand converges slowly with a fixed Gauss grid (see README, "Accuracy").
    # z = 0 is omitted because the on-axis field vanishes there by symmetry.
    zc = np.array([1.4, -1.6, 2.5])
    Enum = field(np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc]), pts, w)[:, 2]
    ref = np.array([cylinder_axis_closed(zi, 1.0, 2.0) for zi in zc])
    print(f"  cylinder  max error: {np.max(np.abs(Enum - ref) / np.abs(ref)):.2e}")

    # Long-cylinder limit: far from the ends the field approaches Gauss's law
    # for an infinite cylinder, E = K * 2 * lam_eff / rho' with lam_eff = Q / L.
    # (Checked on the wire: E_rho -> 2 K lam / rho as L -> infinity.)
    rho = 0.8
    Ew = wire_closed(np.array([[rho, 0.0, 0.0]]), L=2e4, lam=1.0)[0, 0]
    print(f"  wire L->inf Gauss check: {Ew:.6f} vs 2K/rho = {2 * K / rho:.6f}")


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
def _grid_points(center_box: float, n: int, pts: np.ndarray, min_dist: float):
    """Regular grid of query points, dropping those too close to the charge."""
    axis = np.linspace(-center_box, center_box, n)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    dist, _ = cKDTree(pts).query(grid)
    return grid[dist > min_dist]


def build_figure(title: str, geometry: list, pts: np.ndarray, w: np.ndarray,
                 box: float, scale: float, n_grid: int = 7,
                 keep=None) -> go.Figure:
    query = _grid_points(box, n_grid, pts, min_dist=0.25 * scale)
    if keep is not None:
        query = query[keep(query)]
    E = field(query, pts, w)
    mag = np.linalg.norm(E, axis=1)
    unit = E / mag[:, None]
    logm = np.log10(mag)
    lo, hi = np.percentile(logm, [5, 95])

    fig = go.Figure(geometry)
    fig.add_trace(go.Cone(
        x=query[:, 0], y=query[:, 1], z=query[:, 2],
        u=unit[:, 0], v=unit[:, 1], w=unit[:, 2],
        anchor="tail", sizemode="absolute", sizeref=0.22 * scale,
        colorscale="Turbo", cmin=lo, cmax=hi,
        colorbar=dict(title="log10 |E|"), showscale=True,
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-box, box]), yaxis=dict(range=[-box, box]),
            zaxis=dict(range=[-box, box]), aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_dark",
    )
    return fig


def _line(points: np.ndarray, color: str, name: str) -> go.Scatter3d:
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines",
                        line=dict(color=color, width=6), name=name, showlegend=False)


def _disk_surface(R: float, z0: float, color: str) -> go.Surface:
    r = np.linspace(0, R, 20)
    t = np.linspace(0, 2 * np.pi, 60)
    rr, tt = np.meshgrid(r, t)
    return go.Surface(x=rr * np.cos(tt), y=rr * np.sin(tt), z=np.full_like(rr, z0),
                      colorscale=[[0, color], [1, color]], showscale=False, opacity=0.8)


def _cylinder_surface(R: float, L: float, color: str) -> list:
    t = np.linspace(0, 2 * np.pi, 60)
    z = np.linspace(-L / 2, L / 2, 2)
    tt, zz = np.meshgrid(t, z)
    side = go.Surface(x=R * np.cos(tt), y=R * np.sin(tt), z=zz,
                      colorscale=[[0, color], [1, color]], showscale=False, opacity=0.5)
    return [side, _disk_surface(R, L / 2, color), _disk_surface(R, -L / 2, color)]


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
def make_all_figures() -> dict[str, go.Figure]:
    FIG_DIR.mkdir(exist_ok=True)
    figs = {}

    L = 2.0
    pts, w = sample_wire(L)
    figs["wire"] = build_figure(
        "Finite wire, L = 2: field vectors around the segment",
        [_line(np.column_stack([np.zeros(2), np.zeros(2), [-L / 2, L / 2]]), "#ffd166", "wire")],
        pts, w, box=1.8, scale=L)

    R, L = 1.0, 2.0
    pts, w = sample_cylinder(R, L)
    figs["cylinder"] = build_figure(
        "Finite solid cylinder, R = 1, L = 2",
        _cylinder_surface(R, L, "#8ecae6"), pts, w, box=2.2, scale=R,
        # keep exterior points only (interior accuracy is limited, see README)
        keep=lambda q: (np.hypot(q[:, 0], q[:, 1]) > R) | (np.abs(q[:, 2]) > L / 2))

    R = 1.0
    pts, w = sample_ring(R)
    t = np.linspace(0, 2 * np.pi, 200)
    figs["ring"] = build_figure(
        "Thin ring, R = 1: field on and around the ring axis",
        [_line(np.column_stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)]), "#ef476f", "ring")],
        pts, w, box=2.0, scale=R)

    R = 1.0
    pts, w = sample_disk(R)
    figs["disk"] = build_figure(
        "Uniform disk, R = 1: field above and below the plane",
        [_disk_surface(R, 0.0, "#06d6a0")], pts, w, box=2.0, scale=R)

    return figs


def main() -> None:
    run_checks()
    FIG_DIR.mkdir(exist_ok=True)
    for name, fig in make_all_figures().items():
        out = FIG_DIR / f"{name}.html"
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"wrote {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
