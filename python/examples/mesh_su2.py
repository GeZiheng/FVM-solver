"""SU2 grid generators shared by the Phase B application examples."""

import math


def l_shaped_bfs_grid(n):
    """Return an L-shaped backward-facing-step grid as SU2 text.

    The step height and inlet channel height are both one.  ``n`` is the
    number of cells per unit length, so the inlet section is
    ``10*n x n`` cells and the downstream section is ``30*n x 2*n`` cells;
    together they contain ``70*n*n`` cells.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    cells_x = 40 * n
    cells_y = 2 * n
    step_i = 10 * n
    step_j = n

    def full_point_id(i, j):
        return j * (cells_x + 1) + i + 1

    # The lower-left block is absent.  The upper-left loop covers the inlet
    # channel; the lower-right loop covers the full downstream channel.
    cell_loops = []
    for j in range(step_j, cells_y):
        for i in range(step_i):
            cell_loops.append((full_point_id(i, j),
                               full_point_id(i + 1, j),
                               full_point_id(i + 1, j + 1),
                               full_point_id(i, j + 1)))
    for j in range(cells_y):
        for i in range(step_i, cells_x):
            cell_loops.append((full_point_id(i, j),
                               full_point_id(i + 1, j),
                               full_point_id(i + 1, j + 1),
                               full_point_id(i, j + 1)))

    markers = [
        ("inlet", [(full_point_id(0, j), full_point_id(0, j + 1))
                   for j in range(step_j, cells_y)]),
        ("outlet", [(full_point_id(cells_x, j),
                     full_point_id(cells_x, j + 1))
                    for j in range(cells_y)]),
        ("top", [(full_point_id(i, cells_y),
                  full_point_id(i + 1, cells_y))
                 for i in range(cells_x)]),
        ("bottom",
         # Upstream floor y = 1, x < 0.
         [(full_point_id(i, step_j), full_point_id(i + 1, step_j))
          for i in range(step_i)]
         # Step face x = 0, y in [0, 1].
         + [(full_point_id(step_i, j), full_point_id(step_i, j + 1))
            for j in range(step_j)]
         # Downstream floor y = 0.
         + [(full_point_id(i, 0), full_point_id(i + 1, 0))
            for i in range(step_i, cells_x)]),
    ]

    # Interior nodes of the absent lower-left block would otherwise remain
    # unreferenced and appear as points in VTU output.  Compact the 1-based
    # ids while retaining the row-major order of the referenced points.
    used_ids = set()
    for loop in cell_loops:
        used_ids.update(loop)
    for _, faces in markers:
        for first, second in faces:
            used_ids.update((first, second))
    id_map = {full_id: local_id
              for local_id, full_id in enumerate(sorted(used_ids), start=1)}

    lines = ["NDIME= 2", f"NPOIN= {len(id_map)}"]
    for full_id in sorted(id_map):
        i = (full_id - 1) % (cells_x + 1)
        j = (full_id - 1) // (cells_x + 1)
        x = i / n - 10.0
        y = j / n
        lines.append(f"{id_map[full_id]} {x!r} {y!r}")

    lines.append(f"NELEM= {len(cell_loops)}")
    for loop in cell_loops:
        lines.append("9 " + " ".join(str(id_map[point]) for point in loop))

    lines.append(f"NMARK= {len(markers)}")
    for name, faces in markers:
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {len(faces)}")
        for first, second in faces:
            lines.append(f"3 {id_map[first]} {id_map[second]}")

    return "\n".join(lines) + "\n"


def polar_ogrid(n_theta, n_r, r0, r1):
    """Return a stretched polar O-grid around a circular cylinder.

    Radial nodes are distributed geometrically from ``r0`` to ``r1`` and the
    azimuthal direction is periodic.  Marker order is cylinder, inflow (the
    x < 0 half of the outer boundary), and outflow (the x >= 0 half).
    """
    if not isinstance(n_theta, int) or n_theta < 3:
        raise ValueError("n_theta must be an integer >= 3")
    if not isinstance(n_r, int) or n_r <= 0:
        raise ValueError("n_r must be a positive integer")
    if r0 <= 0.0 or r1 <= r0:
        raise ValueError("radii must satisfy 0 < r0 < r1")

    q = (r1 / r0) ** (1.0 / n_r)

    def coordinate(theta_index, radial_index):
        theta = 2.0 * math.pi * theta_index / n_theta
        radius = r0 * q ** radial_index
        return radius * math.cos(theta), radius * math.sin(theta)

    def point_id(theta_index, radial_index):
        return radial_index * n_theta + theta_index + 1

    lines = ["NDIME= 2", f"NPOIN= {n_theta * (n_r + 1)}"]
    for radial_index in range(n_r + 1):
        for theta_index in range(n_theta):
            x, y = coordinate(theta_index, radial_index)
            lines.append(f"{point_id(theta_index, radial_index)} "
                         f"{x!r} {y!r}")

    lines.append(f"NELEM= {n_theta * n_r}")
    for radial_index in range(n_r):
        for theta_index in range(n_theta):
            next_theta = (theta_index + 1) % n_theta
            # Counter-clockwise winding: radial out, azimuth on, radial in.
            lines.append(
                f"9 {point_id(next_theta, radial_index + 1)} "
                f"{point_id(theta_index, radial_index + 1)} "
                f"{point_id(theta_index, radial_index)} "
                f"{point_id(next_theta, radial_index)}")

    inflow = []
    outflow = []
    for theta_index in range(n_theta):
        edge = (point_id(theta_index, n_r),
                point_id((theta_index + 1) % n_theta, n_r))
        theta_mid = 2.0 * math.pi * (theta_index + 0.5) / n_theta
        if math.cos(theta_mid) < 0.0:
            inflow.append(edge)
        else:
            outflow.append(edge)

    markers = [
        ("cylinder", [(point_id(theta_index, 0),
                       point_id((theta_index + 1) % n_theta, 0))
                      for theta_index in range(n_theta)]),
        ("inflow", inflow),
        ("outflow", outflow),
    ]

    lines.append(f"NMARK= {len(markers)}")
    for name, faces in markers:
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {len(faces)}")
        for first, second in faces:
            lines.append(f"3 {first} {second}")

    return "\n".join(lines) + "\n"
