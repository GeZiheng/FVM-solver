"""Pure OpenFOAM ascii dictionary/field parsing and mesh remapping helpers.

This module is deliberately free of WSL, subprocess and file I/O: every
function takes text (or plain lists) and returns plain data, so it can be
unit-tested directly from the repository and reused by later mesh readers.
The MCP server (openfoam_server.py) keeps the WSL reads and assembles the
tool payloads; this module owns everything textual.

Only ascii field files are supported.  Callers detect the binary case via
parse_write_format()/looks_binary() and reject it explicitly instead of
parsing garbage.
"""

import re


# ---------------------------------------------------------------------------
# OpenFOAM dictionary helpers (ascii)
# ---------------------------------------------------------------------------


def strip_foam_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def paren_body(text: str, open_index: int) -> str | None:
    """Body of the parenthesis starting at ``open_index``."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i]
    return None


def block_body(text: str, key: str) -> str | None:
    """Body of the first ``key { ... }`` block."""
    match = re.search(rf"\b{re.escape(key)}\b\s*\{{", text)
    if not match:
        return None
    open_index = match.end() - 1
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i]
    return None


def named_blocks(body: str) -> list[tuple[str, str]]:
    """``name { ... }`` blocks at the top level of ``body``."""
    blocks = []
    index = 0
    while True:
        match = re.compile(r"([A-Za-z_][\w.\-]*)\s*\{").search(body, index)
        if not match:
            return blocks
        open_index = match.end() - 1
        depth = 0
        for i in range(open_index, len(body)):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append((match.group(1), body[open_index + 1 : i]))
                    index = i + 1
                    break
        else:
            return blocks


def parse_entries(kind: str, body: str) -> list:
    if kind == "vector":
        values = []
        for chunk in re.findall(r"\(([^()]*)\)", body):
            parts = [float(v) for v in chunk.split()]
            values.append(parts)
        return values
    return [
        float(v)
        for v in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", body)
    ]


def parse_foam_list(text: str, key: str) -> tuple[str | None, list | None, int | None]:
    """Parse ``key uniform v;`` / ``key nonuniform List<T> N ( ... )``."""
    match = re.search(rf"\b{re.escape(key)}\s+uniform\s+([^;]+);", text)
    if match:
        token = match.group(1).strip()
        try:
            value = (
                [float(token)] if token[0] != "(" else parse_entries("vector", token)[0]
            )
        except (ValueError, IndexError):
            return "uniform", None, None
        return "uniform", value, 1
    match = re.search(
        rf"\b{re.escape(key)}\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", text
    )
    if not match:
        return None, None, None
    count = int(match.group(2))
    body = paren_body(text, match.end() - 1)
    if body is None:
        return "nonuniform", None, count
    return "nonuniform", parse_entries(match.group(1), body), count


def parse_bare_list(text: str, kind: str = "label") -> tuple[list | None, int | None]:
    """polyMesh ``owner``/``neighbour`` style: a bare ``<count> ( ... )`` list
    (no ``internalField`` keyword)."""
    match = re.search(r"\b(\d+)\s*\(", text)
    if not match:
        return None, None
    count = int(match.group(1))
    body = paren_body(text, match.end() - 1)
    if body is None:
        return None, count
    return parse_entries(kind, body), count


def looks_binary(text: str) -> bool:
    """Heuristic binary payload detection for decoded ascii reads."""
    return "\x00" in text or "\ufffd" in text


def parse_write_format(control_dict_text: str) -> str | None:
    """``writeFormat`` keyword of a controlDict, or None when absent."""
    text = strip_foam_comments(control_dict_text)
    match = re.search(r"\bwriteFormat\s+(\w+)\s*;", text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Field summaries and structured-mesh remapping (cell order j*nx + i)
# ---------------------------------------------------------------------------


def min_max(flat: list, is_vector: bool):
    if not flat:
        return None, None
    if is_vector:
        components = list(zip(*flat))
        return [min(c) for c in components], [max(c) for c in components]
    return min(flat), max(flat)


def side_guess(cells: list[tuple[int, int]], nx: int, ny: int) -> str | None:
    if cells and all(i == 0 for i, _ in cells):
        return "West"
    if cells and all(i == nx - 1 for i, _ in cells):
        return "East"
    if cells and all(j == 0 for _, j in cells):
        return "South"
    if cells and all(j == ny - 1 for _, j in cells):
        return "North"
    return None


SIDE_SIGN = {"East": 1, "North": 1, "West": -1, "South": -1}


def parse_poly_mesh(
    owner_text: str, neighbour_text: str, boundary_text: str
) -> dict:
    """Owner/neighbour/boundary connectivity of a polyMesh as plain data.

    Returns a dict with ``owners``, ``neighbours``, ``n_internal``,
    ``n_cells``, ``n_cells_note``, ``patches`` and ``warnings``; the caller
    keeps the fatal nx*ny mismatch decision.
    """
    warnings: list[str] = []
    owners: list[int] = []
    neighbours: list[int] = []
    n_internal = None
    patches: list[dict] = []
    n_cells_note = None
    note = re.search(r"nCells:\s*(\d+)", owner_text)
    if note:
        n_cells_note = int(note.group(1))
    note = re.search(r"nInternalFaces:\s*(\d+)", owner_text)
    if note:
        n_internal = int(note.group(1))
    owner_values, _ = parse_bare_list(strip_foam_comments(owner_text))
    neighbour_values, _ = parse_bare_list(strip_foam_comments(neighbour_text))
    if owner_values is None or neighbour_values is None:
        warnings.append("polyMesh owner/neighbour list not parsed")
        n_cells = None
    else:
        owners = [int(v) for v in owner_values]
        neighbours = [int(v) for v in neighbour_values]
        n_internal = n_internal or len(neighbours)
        # Derive the cell count from the mesh itself: cell labels are
        # contiguous and every cell owns or neighbours at least one face, so
        # max+1 is exact.  The header note (nCells) is only a cross-check -
        # some meshes do not carry it.
        n_cells = max(max(owners, default=-1), max(neighbours, default=-1)) + 1
        if n_cells_note is not None and n_cells_note != n_cells:
            warnings.append(
                f"polyMesh header says nCells: {n_cells_note} but "
                f"owner/neighbour imply {n_cells}; using {n_cells}"
            )
        for name, block in named_blocks(strip_foam_comments(boundary_text)):
            n_faces = re.search(r"nFaces\s+(\d+)\s*;", block)
            start_face = re.search(r"startFace\s+(\d+)\s*;", block)
            kind = re.search(r"type\s+(\w+)\s*;", block)
            if n_faces and start_face:
                patches.append(
                    {
                        "name": name,
                        "type": kind.group(1) if kind else None,
                        "n_faces": int(n_faces.group(1)),
                        "start_face": int(start_face.group(1)),
                    }
                )
    return {
        "owners": owners,
        "neighbours": neighbours,
        "n_internal": n_internal,
        "n_cells": n_cells,
        "n_cells_note": n_cells_note,
        "patches": patches,
        "warnings": warnings,
    }


def build_cell_grids(flat: list, nx: int, ny: int) -> dict:
    """Cell-centred (i, j) grids: ``grid[j][i]`` for scalars and
    ``grid_x``/``grid_y``/``grid_z`` component grids for vectors."""
    grids: dict = {}
    if flat and isinstance(flat[0], list):
        components = list(zip(*flat))
        for comp, letter in enumerate("xyz"):
            if comp < len(components):
                grids[f"grid_{letter}"] = [
                    [components[comp][j * nx + i] for i in range(nx)] for j in range(ny)
                ]
    else:
        grids["grid"] = [[flat[j * nx + i] for i in range(nx)] for j in range(ny)]
    return grids


def map_internal_fluxes(
    phi_values: list,
    owners: list[int],
    neighbours: list[int],
    n_internal: int,
    nx: int,
    ny: int,
) -> tuple[list, list, int]:
    """Internal face fluxes onto x/y face arrays of this project's layout.

    OpenFOAM stores one flux per face (positive owner->neighbour); the output
    arrays use FaceFluxField's +x/+y orientation.  Non-axis-aligned faces are
    counted in ``unmapped`` and left unset.
    """
    x_faces = [[None] * (nx + 1) for _ in range(ny)]
    y_faces = [[None] * nx for _ in range(ny + 1)]
    unmapped = 0
    for face in range(n_internal):
        o = owners[face]
        n = neighbours[face]
        i_o, j_o = o % nx, o // nx
        i_n, j_n = n % nx, n // nx
        value = phi_values[face]
        if j_o == j_n and abs(i_o - i_n) == 1:
            i_face = max(i_o, i_n)
            x_faces[j_o][i_face] = value if i_o < i_n else -value
        elif i_o == i_n and abs(j_o - j_n) == 1:
            j_face = max(j_o, j_n)
            y_faces[j_face][i_o] = value if j_o < j_n else -value
        else:
            unmapped += 1
    return x_faces, y_faces, unmapped


def parse_boundary_patch_values(field_text: str) -> dict:
    """``boundaryField`` patch ``value`` entries of a field file."""
    patch_values: dict = {}
    boundary_body = block_body(strip_foam_comments(field_text), "boundaryField")
    if boundary_body is not None:
        for name, block in named_blocks(boundary_body):
            kind, vals, count = parse_foam_list(block, "value")
            patch_values[name] = {"value_kind": kind, "count": count, "values": vals}
    return patch_values


def build_boundary_patches(
    mesh_patches: list[dict],
    patch_values: dict,
    owners: list[int],
    nx: int,
    ny: int,
    max_vals: int,
) -> list[dict]:
    """Per-patch owner cells, values and a side guess for a face field."""
    patches = []
    for mesh_patch in mesh_patches:
        start = mesh_patch["start_face"]
        n_faces = mesh_patch["n_faces"]
        cells = [
            (owners[f] % nx, owners[f] // nx)
            for f in range(start, min(start + n_faces, len(owners)))
        ]
        info = patch_values.get(mesh_patch["name"], {})
        side = side_guess(cells, nx, ny)
        patch_entry = {
            "name": mesh_patch["name"],
            "type": mesh_patch["type"],
            "n_faces": n_faces,
            "side": side,
            "owner_cells": [list(c) for c in cells[:max_vals]],
            "values": (info.get("values") or [])[:max_vals],
            "value_kind": info.get("value_kind"),
        }
        if side in SIDE_SIGN:
            patch_entry["our_sign"] = SIDE_SIGN[side]
            patch_entry["values_ours"] = [
                SIDE_SIGN[side] * v for v in (info.get("values") or [])[:max_vals]
            ]
        if (info.get("values") is not None) and len(info["values"]) > max_vals:
            patch_entry["truncated"] = True
        patches.append(patch_entry)
    return patches
