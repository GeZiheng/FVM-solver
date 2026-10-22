"""Pressure-driven flow in a converging (trapezoidal) channel via PISO.

The trapezoidal quad mesh is non-orthogonal (the x-faces tilt with the
side walls), which exercises the 4.3c non-orthogonal diffusion correction
and skew-corrected face interpolation.  The same mesh is reproduced by an
OpenFOAM blockMesh case for the cross-check documented in
docs/numerical.md (single step, endTime == dt).

Run:  python python/examples/piso_nonortho_channel.py [--correctors N] [--no-corrections]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: F401,E402

import numpy as np
import pyfvm as fvm

# Bilinear map of the unit square onto the trapezoid: the top edge is
# narrowed to [0.3, 0.7], so the side walls (and with them the x-faces)
# tilt by atan(0.3) ~ 16.7 degrees.
N = 16
CORNERS = ((0.0, 0.0), (1.0, 0.0), (0.7, 1.0), (0.3, 1.0))


def point(i, j):
    s = i / N
    t = j / N
    x = (1.0 - t) * ((1.0 - s) * CORNERS[0][0] + s * CORNERS[1][0]) \
        + t * ((1.0 - s) * CORNERS[3][0] + s * CORNERS[2][0])
    y = (1.0 - t) * ((1.0 - s) * CORNERS[0][1] + s * CORNERS[1][1]) \
        + t * ((1.0 - s) * CORNERS[3][1] + s * CORNERS[2][1])
    return x, y


def su2_text():
    lines = ["NDIME= 2", f"NPOIN= {(N + 1) * (N + 1)}"]
    pid = lambda i, j: j * (N + 1) + i + 1
    for j in range(N + 1):
        for i in range(N + 1):
            x, y = point(i, j)
            lines.append(f"{pid(i, j)} {x!r} {y!r}")
    lines.append(f"NELEM= {N * N}")
    for j in range(N):
        for i in range(N):
            lines.append(f"9 {pid(i, j)} {pid(i + 1, j)} "
                         f"{pid(i + 1, j + 1)} {pid(i, j + 1)}")
    lines.append("NMARK= 4")
    markers = [
        ("south", [(pid(i, 0), pid(i + 1, 0)) for i in range(N)]),
        ("east", [(pid(N, j), pid(N, j + 1)) for j in range(N)]),
        ("north", [(pid(i + 1, N), pid(i, N)) for i in range(N)]),
        ("west", [(pid(0, j + 1), pid(0, j)) for j in range(N)]),
    ]
    for name, faces in markers:
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {len(faces)}")
        for a, b in faces:
            lines.append(f"3 {a} {b}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctors", type=int, default=2)
    parser.add_argument("--no-corrections", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trapezoid.su2")
        with open(path, "w", encoding="ascii") as fh:
            fh.write(su2_text())
        mesh = fvm.read_su2(path)

    # Patch order follows the marker order: 0 south, 1 east, 2 north, 3 west.
    velocity = fvm.VectorField(mesh, "velocity")
    pressure = fvm.ScalarField(mesh, "pressure")
    flux = fvm.FaceFluxField(mesh, "phi")

    bc_u = fvm.BoundaryField(mesh)
    bc_v = fvm.BoundaryField(mesh)
    bc_p = fvm.BoundaryField(mesh)
    for wall in (0, 2):  # no-slip walls
        bc_u.set(wall, fvm.BCType.Dirichlet, 0.0)
        bc_v.set(wall, fvm.BCType.Dirichlet, 0.0)
        bc_p.set(wall, fvm.BCType.Neumann, 0.0)
    bc_u.set(1, fvm.BCType.Neumann, 0.0)   # outlet
    bc_v.set(1, fvm.BCType.Neumann, 0.0)
    bc_p.set(1, fvm.BCType.Dirichlet, 0.0)
    bc_u.set(3, fvm.BCType.Neumann, 0.0)   # inlet
    bc_v.set(3, fvm.BCType.Neumann, 0.0)
    bc_p.set(3, fvm.BCType.Dirichlet, 1.0)

    cfg = fvm.PisoConfig()
    cfg.dt = 0.02
    cfg.n_steps = 1
    cfg.n_correctors = args.correctors
    cfg.scheme = fvm.ConvectionScheme.Upwind
    cfg.time_scheme = fvm.TimeScheme.Euler
    cfg.solver_config.tolerance = 1e-9
    cfg.solver_config.max_iterations = 20000
    if args.no_corrections:
        cfg.corrections.non_orthogonal_diffusion = False
        cfg.corrections.skew_interpolation = False
        cfg.corrections.n_non_orthogonal_correctors = 0

    result = fvm.solve_piso(mesh, 1.0, 1.0, bc_u, bc_v, bc_p, cfg,
                            velocity, pressure, flux)

    phi = flux.to_numpy()
    inlet_faces = mesh.patch_faces(3)
    inlet_length = float(np.hypot(CORNERS[3][0] - CORNERS[0][0],
                                  CORNERS[3][1] - CORNERS[0][1]))
    inlet_flux = float(np.sum(phi[inlet_faces]))
    print(f"correctors={cfg.n_correctors} "
          f"corrections={'off' if args.no_corrections else 'on'}")
    print(f"p(0,0) = {pressure.to_numpy()[0]:.10e}")
    print(f"inlet phi/length = {abs(inlet_flux) / inlet_length:.10e}")
    print(f"continuity = {result.history[-1].continuity:.3e}")

    suffix = "off" if args.no_corrections else "on"
    output_dir = Path(__file__).resolve().parent / "output" / "nonortho_channel"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nonortho_channel_nc{cfg.n_correctors}_{suffix}.vtu"
    fvm.write_vtu(str(output_path), mesh,
                  {"pressure": pressure}, {"velocity": velocity})
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

