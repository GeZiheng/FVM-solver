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


class _PointPool:
    """Deduplicating pool of SU2 points keyed by exact coordinates.

    Multiblock generators share edge points between adjacent blocks; the
    shared points are computed once and re-added, so exact float equality
    identifies them.
    """

    def __init__(self):
        self._ids = {}
        self.points = []

    def add(self, x, y):
        key = (x, y)
        point_id = self._ids.get(key)
        if point_id is None:
            self.points.append(key)
            point_id = len(self.points)
            self._ids[key] = point_id
        return point_id

    def id_of(self, x, y):
        """Existing id for the coordinates; raises on unknown points."""
        try:
            return self._ids[(x, y)]
        except KeyError:
            raise KeyError(f"point ({x!r}, {y!r}) not in the pool") from None


def _geometric_intervals(length, n, first):
    """Cumulative station list [0, length] of n geometric intervals.

    Interval sizes grow by a common ratio from ``first`` at the start; the
    ratio is solved by bisection so the total matches ``length`` exactly
    (the final station is pinned).  Falls back to a uniform distribution
    when ``n * first`` already exceeds ``length``.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if first <= 0.0:
        raise ValueError("first must be positive")
    if n * first >= length:
        return [length * i / n for i in range(n + 1)]

    def total(ratio):
        return first * (ratio ** n - 1.0) / (ratio - 1.0) if ratio != 1.0 \
            else first * n

    low, high = 1.0, 2.0
    while total(high) < length:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if total(mid) < length:
            low = mid
        else:
            high = mid
    ratio = 0.5 * (low + high)
    stations = [0.0]
    for i in range(1, n + 1):
        stations.append(first * (ratio ** i - 1.0) / (ratio - 1.0)
                        if i < n else length)
    return stations


def _stretched_axis(near, far, n, first):
    """Ascending stations between the ``near`` (fine) and ``far`` ends.

    Both endpoints are pinned exactly -- the fine end through a zero
    first interval, the far end by overwriting the boundary station
    (the last for an ascending axis, the first for a descending one) --
    so block boundaries on the domain sides deduplicate in the pool.
    """
    intervals = _geometric_intervals(abs(far - near), n, first)
    if near < far:
        axis = [near + d for d in intervals]
        axis[-1] = far
        return axis
    axis = [near - d for d in reversed(intervals)]
    axis[0] = far
    return axis


def _tensor_block(pool, xs, ys):
    """Emit the quad cells of a tensor block over the given axis stations.

    ``xs`` and ``ys`` must be ascending; cells are wound counter-clockwise
    (i, j), (i+1, j), (i+1, j+1), (i, j+1).  Points shared with adjacent
    blocks resolve to their existing pool ids.
    """
    ids = [[pool.add(x, y) for y in ys] for x in xs]
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cells.append((ids[i][j], ids[i + 1][j],
                          ids[i + 1][j + 1], ids[i][j + 1]))
    return cells


def cylinder_rectangle_grid(n_theta=96, n_r=40, radius=0.5, half=3.0,
                            domain=(-10.0, 25.0, -10.0, 10.0),
                            n_w=24, n_e=36, n_n=24, n_s=24):
    """Return an O-4 multiblock cylinder grid in a rectangular domain.

    Topology (classic square-in-rectangle): a polar core connects the
    circle of ``radius`` to the square ``[-half, half]^2`` -- uniform
    azimuths with per-ray geometric radial stretching to R(theta) =
    half / max(|cos|, |sin|), the 45-degree rays landing exactly on the
    square corners (the O-4 split).  Eight axis-aligned tensor blocks
    (four strips plus four corners) extend the square edges to the
    ``domain`` rectangle ``(x_min, x_max, y_min, y_max)``; their 1-D
    stations are geometric with the first interval ``half*d_theta`` (the
    square-edge transverse spacing), refined towards the square.  All
    inter-block faces are point-conforming via a shared coordinate pool.

    ``n_theta`` must be a multiple of 8 (45-degree rays on grid lines);
    each square edge carries ``n_theta / 4`` cells.  Marker order is
    cylinder, inlet (west), outlet (east), top, bottom.
    """
    x_min, x_max, y_min, y_max = domain
    integer_args = {"n_theta": n_theta, "n_r": n_r, "n_w": n_w,
                    "n_e": n_e, "n_n": n_n, "n_s": n_s}
    for name, value in integer_args.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_theta % 8 != 0:
        raise ValueError("n_theta must be a multiple of 8")
    if not 0.0 < radius < half:
        raise ValueError("radii must satisfy 0 < radius < half")
    if not x_min < -half or not half < x_max \
            or not y_min < -half or not half < y_max:
        raise ValueError("domain must strictly contain the square")

    d_theta = 2.0 * math.pi / n_theta
    pool = _PointPool()
    cells = []

    # Polar core: point ids [theta_index][radial_index].  The outermost
    # ring is snapped onto the square analytically -- the 45-degree rays
    # (i = eighth + m*quarter) hit the corners exactly, everything else
    # lands on the edge via tan -- so its floats match the tensor-block
    # corners and the shared pool deduplicates them.
    eighth = n_theta // 8
    quarter = n_theta // 4
    core = []
    for i in range(n_theta):
        theta = d_theta * i
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        outer = half / max(abs(cos_t), abs(sin_t))
        ratio = (outer / radius) ** (1.0 / n_r)
        ring = []
        for k in range(n_r + 1):
            if k < n_r:
                r = radius * ratio ** k
                ring.append(pool.add(r * cos_t, r * sin_t))
            elif (i - eighth) % quarter == 0:
                ring.append(pool.add(half if cos_t > 0.0 else -half,
                                     half if sin_t > 0.0 else -half))
            elif abs(cos_t) >= abs(sin_t):
                x = half if cos_t > 0.0 else -half
                ring.append(pool.add(x, x * sin_t / cos_t))
            else:
                y = half if sin_t > 0.0 else -half
                ring.append(pool.add(y * cos_t / sin_t, y))
        core.append(ring)
    for k in range(n_r):
        for i in range(n_theta):
            # Same winding as the retired polar O-grid (reader normalizes).
            cells.append((core[(i + 1) % n_theta][k + 1], core[i][k + 1],
                          core[i][k], core[(i + 1) % n_theta][k]))

    # Square-edge point coordinates in ascending order, reused by the
    # strips so the block interfaces are exactly conforming.
    north_x = [pool.points[core[i][n_r] - 1][0]
               for i in range(3 * eighth, eighth - 1, -1)]
    east_y = [pool.points[core[i % n_theta][n_r] - 1][1]
              for i in range(7 * eighth, 9 * eighth + 1)]
    south_x = [pool.points[core[i][n_r] - 1][0]
               for i in range(5 * eighth, 7 * eighth + 1)]
    west_y = [pool.points[core[i][n_r] - 1][1]
              for i in range(5 * eighth, 3 * eighth - 1, -1)]

    first = half * d_theta
    west_x = _stretched_axis(-half, x_min, n_w, first)
    east_x = _stretched_axis(half, x_max, n_e, first)
    north_y = _stretched_axis(half, y_max, n_n, first)
    south_y = _stretched_axis(-half, y_min, n_s, first)

    for xs, ys in ((west_x, north_y), (east_x, north_y),
                   (west_x, south_y), (east_x, south_y),
                   (west_x, west_y), (east_x, east_y),
                   (north_x, north_y), (south_x, south_y)):
        cells.extend(_tensor_block(pool, xs, ys))

    markers = [
        ("cylinder", [(core[i][0], core[(i + 1) % n_theta][0])
                      for i in range(n_theta)]),
        ("inlet", _boundary_edges(pool, south_y + west_y[1:] + north_y[1:],
                                  vertical=True, fixed=x_min)),
        ("outlet", _boundary_edges(pool, south_y + east_y[1:] + north_y[1:],
                                   vertical=True, fixed=x_max)),
        ("top", _boundary_edges(pool, west_x + north_x[1:] + east_x[1:],
                                vertical=False, fixed=y_max)),
        ("bottom", _boundary_edges(pool, west_x + south_x[1:] + east_x[1:],
                                   vertical=False, fixed=y_min)),
    ]

    lines = ["NDIME= 2", f"NPOIN= {len(pool.points)}"]
    for point_id, (x, y) in enumerate(pool.points, start=1):
        lines.append(f"{point_id} {x!r} {y!r}")
    lines.append(f"NELEM= {len(cells)}")
    for loop in cells:
        lines.append("9 " + " ".join(str(point) for point in loop))
    lines.append(f"NMARK= {len(markers)}")
    for name, faces in markers:
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {len(faces)}")
        for first_id, second_id in faces:
            lines.append(f"3 {first_id} {second_id}")
    return "\n".join(lines) + "\n"


def _boundary_edges(pool, stations, vertical, fixed):
    """Marker edge id pairs along one domain side, stations ascending.

    The side points must already exist in the pool (tensor-block
    boundaries); unknown coordinates raise instead of minting dangling
    points.  ``vertical`` sides hold ``fixed`` x with ``stations`` as y
    values, horizontal sides the transpose.
    """
    if vertical:
        ids = [pool.id_of(fixed, value) for value in stations]
    else:
        ids = [pool.id_of(value, fixed) for value in stations]
    return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]


def sheared_channel(n, shear=0.3):
    """Return a sheared constant-section channel as SU2 text.

    The unit square is mapped bilinearly onto the parallelogram with
    corners (0, 0), (1, 0), (1 + shear, 1), (shear, 1): the south and
    north edges stay horizontal (constant section height one) while the
    x-faces tilt by atan(shear), which exercises the non-orthogonal
    corrections.  ``n`` is the cell count per edge; the grid is
    ``n x n`` quads.  Marker order is south, east, north, west
    (patch 0..3), matching piso_nonortho_channel.py.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if shear <= 0.0:
        raise ValueError("shear must be positive")

    corners = ((0.0, 0.0), (1.0, 0.0), (1.0 + shear, 1.0), (shear, 1.0))

    def point(i, j):
        s = i / n
        t = j / n
        x = (1.0 - t) * ((1.0 - s) * corners[0][0] + s * corners[1][0]) \
            + t * ((1.0 - s) * corners[3][0] + s * corners[2][0])
        y = (1.0 - t) * ((1.0 - s) * corners[0][1] + s * corners[1][1]) \
            + t * ((1.0 - s) * corners[3][1] + s * corners[2][1])
        return x, y

    def point_id(i, j):
        return j * (n + 1) + i + 1

    lines = ["NDIME= 2", f"NPOIN= {(n + 1) * (n + 1)}"]
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = point(i, j)
            lines.append(f"{point_id(i, j)} {x!r} {y!r}")

    lines.append(f"NELEM= {n * n}")
    for j in range(n):
        for i in range(n):
            lines.append(f"9 {point_id(i, j)} {point_id(i + 1, j)} "
                         f"{point_id(i + 1, j + 1)} {point_id(i, j + 1)}")

    markers = [
        ("south", [(point_id(i, 0), point_id(i + 1, 0)) for i in range(n)]),
        ("east", [(point_id(n, j), point_id(n, j + 1)) for j in range(n)]),
        ("north", [(point_id(i + 1, n), point_id(i, n)) for i in range(n)]),
        ("west", [(point_id(0, j + 1), point_id(0, j)) for j in range(n)]),
    ]
    lines.append(f"NMARK= {len(markers)}")
    for name, faces in markers:
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {len(faces)}")
        for first, second in faces:
            lines.append(f"3 {first} {second}")

    return "\n".join(lines) + "\n"
