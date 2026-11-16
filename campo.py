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
# Visualization (style: "Slate Mint")
# --------------------------------------------------------------------------
BG = "#1b262c"
GRID = "#3d5361"
TEXT = "#bbe1fa"
CHARGE = "#00ffab"
FIELD_SCALE = [[0.0, "#2f7fb5"], [0.5, "#6fb6e6"], [1.0, "#e3f4ff"]]


def shell_points(radii, z_levels, n_phi: int = 8, axis: bool = True) -> np.ndarray:
    """Query points on cylindrical shells around the z axis, plus the axis itself."""
    # half-step offset so no arrow points straight at the default camera
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False) + np.pi / n_phi
    R, P, Z = np.meshgrid(radii, phi, z_levels, indexing="ij")
    out = np.column_stack([(R * np.cos(P)).ravel(), (R * np.sin(P)).ravel(), Z.ravel()])
    if axis:
        zl = np.asarray(z_levels, dtype=float)
        out = np.vstack([out, np.column_stack([np.zeros_like(zl), np.zeros_like(zl), zl])])
    return out


def arrow_traces(query: np.ndarray, E: np.ndarray, length: float) -> list:
    """Thin arrows (line shaft + small cone head), coloured by log10 |E|."""
    mag = np.linalg.norm(E, axis=1)
    unit = E / mag[:, None]
    logm = np.log10(mag)
    lo, hi = np.percentile(logm, [3, 97])
    t = np.clip((logm - lo) / (hi - lo), 0, 1)
    tip = query + unit * (length * (0.6 + 0.4 * t))[:, None]

    nan = np.full(len(query), np.nan)
    xs, ys, zs = (np.column_stack([query[:, i], tip[:, i], nan]).ravel() for i in range(3))
    shaft = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=np.repeat(logm, 3), colorscale=FIELD_SCALE, cmin=lo, cmax=hi,
                  width=5, showscale=True,
                  colorbar=dict(title=dict(text="log10 |E|", font=dict(color=TEXT)),
                                tickfont=dict(color=TEXT), len=0.6, thickness=14)))
    # Plotly colours cones by the vector norm, so the norm (1 + t) carries the colour.
    v = unit * (1.0 + t)[:, None]
    heads = go.Cone(
        x=tip[:, 0], y=tip[:, 1], z=tip[:, 2], u=v[:, 0], v=v[:, 1], w=v[:, 2],
        anchor="tail", sizemode="scaled", sizeref=0.35, showscale=False,
        colorscale=FIELD_SCALE, cmin=1.0, cmax=2.0, hoverinfo="skip",
        lighting=dict(ambient=0.9, diffuse=0.3))
    return [shaft, heads]


def build_figure(title: str, geometry: list, pts: np.ndarray, w: np.ndarray,
                 query: np.ndarray, box: float, min_dist: float = 0.2,
                 arrow: float = 0.3) -> go.Figure:
    dist, _ = cKDTree(pts).query(query)
    query = query[dist > min_dist]
    E = field(query, pts, w)
    ok = np.linalg.norm(E, axis=1) > 1e-9      # drop exact zeros (e.g. ring centre)
    fig = go.Figure(geometry + arrow_traces(query[ok], E[ok], arrow))
    axis = dict(range=[-box, box], backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID,
                showticklabels=False, title="", showspikes=False)
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=20), x=0.5),
        paper_bgcolor=BG, font=dict(color=TEXT),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
                   camera=dict(eye=dict(x=1.25, y=1.25, z=0.65))),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _line(points: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines",
                        line=dict(color=CHARGE, width=12), showlegend=False,
                        hoverinfo="skip")


def _disk_surface(R: float, z0: float, opacity: float = 0.85) -> go.Surface:
    r = np.linspace(0, R, 20)
    t = np.linspace(0, 2 * np.pi, 60)
    rr, tt = np.meshgrid(r, t)
    return go.Surface(x=rr * np.cos(tt), y=rr * np.sin(tt), z=np.full_like(rr, z0),
                      colorscale=[[0, CHARGE], [1, CHARGE]], showscale=False,
                      opacity=opacity, hoverinfo="skip")


def _cylinder_surface(R: float, L: float) -> list:
    t = np.linspace(0, 2 * np.pi, 60)
    z = np.linspace(-L / 2, L / 2, 2)
    tt, zz = np.meshgrid(t, z)
    side = go.Surface(x=R * np.cos(tt), y=R * np.sin(tt), z=zz,
                      colorscale=[[0, CHARGE], [1, CHARGE]], showscale=False,
                      opacity=0.55, hoverinfo="skip")
    return [side, _disk_surface(R, L / 2, 0.55), _disk_surface(R, -L / 2, 0.55)]


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
def make_all_figures() -> dict[str, go.Figure]:
    figs = {}

    L = 2.0
    pts, w = sample_wire(L)
    figs["wire"] = build_figure(
        "Finite wire (L = 2)",
        [_line(np.array([[0, 0, -L / 2], [0, 0, L / 2]]))], pts, w,
        shell_points([0.5, 1.2], np.linspace(-1.5, 1.5, 7)), box=1.8)

    R, L = 1.0, 2.0
    pts, w = sample_cylinder(R, L)
    q = shell_points([0.5, 1.6], np.linspace(-2.0, 2.0, 7))
    # exterior points only (interior accuracy is limited, see README)
    q = q[(np.hypot(q[:, 0], q[:, 1]) > R) | (np.abs(q[:, 2]) > L / 2)]
    figs["cylinder"] = build_figure(
        "Finite solid cylinder (R = 1, L = 2)",
        _cylinder_surface(R, L), pts, w, q, box=2.3, arrow=0.38)

    R = 1.0
    pts, w = sample_ring(R)
    t = np.linspace(0, 2 * np.pi, 200)
    figs["ring"] = build_figure(
        "Thin ring (R = 1)",
        [_line(np.column_stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)]))], pts, w,
        shell_points([0.45, 1.6], np.linspace(-1.5, 1.5, 7)), box=2.0)

    pts, w = sample_disk(R)
    figs["disk"] = build_figure(
        "Uniform disk (R = 1)",
        [_disk_surface(R, 0.0)], pts, w,
        shell_points([0.5, 1.6], np.linspace(-1.5, 1.5, 7)), box=2.0)

    return figs


def main() -> None:
    run_checks()
    FIG_DIR.mkdir(exist_ok=True)
    for name, fig in make_all_figures().items():
        fig.write_html(FIG_DIR / f"{name}.html", include_plotlyjs="cdn")
        try:  # static screenshot for the README (needs kaleido + Chrome)
            fig.write_image(FIG_DIR / f"{name}.png", width=900, height=760, scale=2)
        except Exception as exc:  # noqa: BLE001
            print(f"  (PNG skipped for {name}: {exc})")
        print(f"wrote figures/{name}.html")


if __name__ == "__main__":
    main()
