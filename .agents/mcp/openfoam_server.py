#!/usr/bin/env python3
"""Stdio MCP server: the single entry point for OpenFOAM cross-checks (WSL).

Speaks the same minimal subset of the Model Context Protocol as
``build_and_test_server.py`` (newline-delimited JSON-RPC 2.0 over
stdin/stdout) and uses only the Python standard library.

Shared by Codex (registered in ``.codex/config.toml``) and opencode (reached
through the ``mcp`` block in ``opencode.json``).  Because MCP servers run
outside the agent sandbox, this removes the per-run escalation that shelling
into WSL from an agent requires, and gives both agents one structured
interface instead of hand-rolled commands.

Tools (all WSL paths; the case root defaults to ``~/OpenFOAM/gzh1057-14/run``
and the OpenFOAM source tree to ``~/OpenFOAM/OpenFOAM-14``):

* ``run_case``      - build a fresh, timestamped case dir and start
                      ``foamRun`` in the background (returns immediately)
* ``run_status``    - poll a background run (state + partial/complete summary)
* ``summarize_log`` - parse a solver log into a per-step table
* ``list_cases``    - list existing case directories
* ``extract_fields``- read a time directory's fields (cell grids and face
                      fluxes, mapped to this project's ``(i,j)`` layout)
* ``find_source``   - grep the OpenFOAM-14 source tree
* ``read_source``   - read a numbered snippet of an OpenFOAM/source/case file

Scope is deliberately narrow: the only mutation is creating a new case
directory and running ``foamRun`` in it.  There is no tool for arbitrary
commands and nothing deletes or overwrites existing cases.

Operating discipline for cross-checks (results belong in ``docs/numerical.md``,
not in this server):

* fix the comparison contract first - mesh, properties, time scheme,
  convection scheme, every BC, dt, solver tolerances, and the algorithm
  settings (``nCorrectors``, relaxation);
* compare a **single step** first (``end_time == dt``) and check ``p(0,0)``
  and the inlet ``phi/S`` to solver-tolerance level before running longer;
* one configuration per case directory - the configuration is part of the
  directory name, never reuse a directory with different settings;
* run both ``nCorrectors = 1`` and ``= 2``: the second p solve's initial
  residual (``res_p`` in ``summarize_log``) proves the corrector is advancing;
* never loosen a test tolerance to hide a difference; explain it with data or
  source instead.

The helper scripts the server drives stay in ``.agents/mcp/openfoam/``:
``make_of_case.sh`` (build + run) and ``of_log_summary.py`` (log -> JSON).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

# Sibling module (same directory): importable because the server script's
# own directory is sys.path[0] when launched as a script.
import openfoam_fields as of_fields

MAX_OUTPUT_CHARS = 8000
SERVER_NAME = "openfoam"
SERVER_VERSION = "1.1.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

MAX_TIMEOUT = 3600

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OF_SCRIPTS = REPO_ROOT / ".agents" / "mcp" / "openfoam"

# Bash expression, not a literal path: it must expand inside WSL.
DEFAULT_ROOT_BASH = '"$HOME/OpenFOAM/gzh1057-14/run"'
DEFAULT_SOLVER = "incompressibleFluid"

# Extra case roots run_case may read templates from and write cases to
# (":"-separated absolute WSL paths); the default run root above is always
# allowed.  Escape hatch for humans, confinement boundary for agents.
ALLOWED_ROOTS_ENV = "FVM_OF_ALLOWED_ROOTS"

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SOLVER_RE = re.compile(r"^[A-Za-z0-9_]+$")
# <name>_nc<N>[_dt<D>][_t<T>]_<MMDD-HHMMSS>, as produced by make_of_case.sh.
CASE_RE = re.compile(
    r"^(?P<name>.+)_nc(?P<nc>\d+)"
    r"(?:_dt(?P<dt>[0-9.eE+-]+))?"
    r"(?:_t(?P<t>[0-9.eE+-]+))?"
    r"_(?P<stamp>\d{4}-\d{6})$"
)

RUN_CASE_DESCRIPTION = (
    "Create a fresh OpenFOAM case in WSL from an existing template, patch its "
    "dictionaries (endTime/deltaT/writeInterval, nCorrectors) and start "
    "foamRun in the background. Returns immediately with case_dir, log_path "
    "and status='running'; poll run_status until the state is terminal "
    "('finished', 'failed' or 'stale'), then read the fields with "
    "extract_fields. Each configuration gets its own "
    "timestamped directory with the configuration baked into the name; "
    "nothing is overwritten. For a cross-check: compare a single step first "
    "(end_time = dt) and run both nCorrectors = 1 and 2."
)

RUN_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "template": {
            "type": "string",
            "description": "Existing case directory to copy (WSL path; must "
            "stay inside an allowed OpenFOAM root - the default run root "
            "plus FVM_OF_ALLOWED_ROOTS entries).",
        },
        "name": {
            "type": "string",
            "description": "Directory name prefix; [A-Za-z0-9._-] only.",
        },
        "n_correctors": {
            "type": "integer",
            "minimum": 1,
            "description": "PISO pressure correctors (2 or more for stability).",
        },
        "dt": {"type": "number", "description": "deltaT (default: template value)."},
        "end_time": {"type": "number", "description": "endTime (default: template value)."},
        "dx": {
            "type": "number",
            "description": "Cell size, needed to convert Co_max into max|U|.",
        },
        "solver": {
            "type": "string",
            "default": DEFAULT_SOLVER,
            "description": "foamRun solver name.",
        },
        "root": {
            "type": "string",
            "description": "Case root (WSL path; default "
            "~/OpenFOAM/gzh1057-14/run). A non-default root must stay inside "
            "an allowed root (default run root plus FVM_OF_ALLOWED_ROOTS).",
        },
        "timeout_s": {
            "type": "integer",
            "description": "Wall-clock timeout for the (fast) case setup only; "
            "the foamRun itself runs in the background (default 120).",
        },
    },
    "required": ["template", "name", "n_correctors"],
    "additionalProperties": False,
}

RUN_STATUS_DESCRIPTION = (
    "Poll a background run started by run_case. Reports state ('running', "
    "'finished', 'failed', 'stale' or 'unknown') from the case's .exitcode "
    "marker and the log; 'stale' means the launcher died before writing "
    ".exitcode, e.g. after a server restart. Includes a per-step summary of "
    "log.run (partial while running) when the log exists. dt/dx default to "
    "the values recorded in the case's mcp_run file at launch."
)

RUN_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "case_dir": {"type": "string", "description": "Case directory in WSL (absolute)."},
        "dt": {"type": "number", "description": "deltaT, for Co -> max|U| (default: mcp_run)."},
        "dx": {"type": "number", "description": "Cell size, for Co -> max|U| (default: mcp_run)."},
    },
    "required": ["case_dir"],
    "additionalProperties": False,
}

SUMMARIZE_LOG_DESCRIPTION = (
    "Parse an OpenFOAM solver log into a per-step JSON table: Courant number, "
    "max|U| (when dt and dx are given), linear-solver initial residuals per "
    "field, continuity error, and a divergence flag. Handles OpenFOAM's line "
    "ordering (the Courant line precedes the step it belongs to) so the step "
    "indices stay consistent."
)

SUMMARIZE_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "log": {"type": "string", "description": "Log file path (WSL path)."},
        "dt": {"type": "number", "description": "deltaT, for Co -> max|U|."},
        "dx": {"type": "number", "description": "Cell size, for Co -> max|U|."},
    },
    "required": ["log"],
    "additionalProperties": False,
}

LIST_CASES_DESCRIPTION = (
    "List OpenFOAM case directories under the case root, newest first, with "
    "the configuration parsed back out of the directory name (nCorrectors, "
    "deltaT, endTime) and whether a log.run is present."
)

LIST_CASES_SCHEMA = {
    "type": "object",
    "properties": {
        "root": {"type": "string", "description": "Case root (WSL path)."},
        "name_filter": {"type": "string", "description": "Substring filter on the directory name."},
        "limit": {"type": "integer", "minimum": 1, "description": "Max entries (default 20)."},
    },
    "required": [],
    "additionalProperties": False,
}


def _tail(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        f"... (truncated, showing last {MAX_OUTPUT_CHARS} chars)\n"
        + text[-MAX_OUTPUT_CHARS:]
    )


def _run_parts(cmd: list[str], timeout: int) -> tuple[int | None, str, str]:
    """Run a command; returns (exit_code, stdout, stderr).

    stdout and stderr stay separate on purpose: ``wsl.exe`` prints localisation
    warnings on stderr, and mixing them into stdout corrupts the JSON the tools
    parse back. ``exit_code`` is None when the command could not be started or
    timed out (the reason is then in stderr).
    """
    try:
        # WSL_UTF8 makes wsl.exe emit UTF-8 instead of UTF-16 (its localisation
        # warnings are otherwise unreadable once decoded as UTF-8).
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env={**os.environ, "WSL_UTF8": "1"},
        )
    except FileNotFoundError:
        return None, "", f"command not found: {cmd[0]} (is WSL installed?)"
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout}s"

    return completed.returncode, (completed.stdout or "").strip(), (completed.stderr or "").strip()


def _combine(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr


def _loads(text: str) -> dict:
    """Parse JSON, tolerating stray lines around the object (e.g. WSL warnings)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _to_wsl(path: pathlib.Path | str) -> str:
    """Map a Windows path (D:\\dir\\file) to its WSL mount (/mnt/d/dir/file)."""
    text = str(path).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", text)
    if not match:
        raise ValueError(f"not a drive-letter path, cannot map into WSL: {path}")
    return f"/mnt/{match.group(1).lower()}/{match.group(2)}"


def _bash(
    words: list[str], extra_args: list[str], timeout: int
) -> tuple[int | None, str, str]:
    """Run a bash command in WSL.

    ``words`` is the command itself (each word gets shell-quoted); ``extra_args``
    are appended unquoted to the argv of the inner bash (they become $1, $2, ...
    in the command, so no interpolation happens).
    """
    command = " ".join(shlex.quote(word) for word in words)
    cmd = ["wsl.exe", "-e", "bash", "-lc", command]
    if extra_args:
        cmd += ["_"] + extra_args
    return _run_parts(cmd, timeout)


def _error(message: str, **extra) -> tuple[str, bool]:
    payload = {"error": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, indent=2), False


def _as_json(payload: dict) -> tuple[str, bool]:
    return json.dumps(payload, ensure_ascii=False, indent=2), True


def _check_root(root: str | None) -> str | None:
    """Returns an error string, or None when the root looks acceptable."""
    if root is None:
        return None
    if not root.startswith("/"):
        return f"root must be an absolute WSL path, got: {root}"
    if "\n" in root or "\r" in root:
        return "root must not contain newlines"
    if root != root.strip():
        return "root must not have leading/trailing whitespace"
    return None


def _within_allowed(resolved_path: str, resolved_roots: list[str]) -> bool:
    """True when ``resolved_path`` equals a root or lies below it."""
    return any(
        root == "/"
        or (root and (resolved_path == root or resolved_path.startswith(root + "/")))
        for root in resolved_roots
    )


def _extra_allowed_roots() -> list[str]:
    """Absolute WSL paths from FVM_OF_ALLOWED_ROOTS (others are ignored)."""
    return [
        path
        for path in os.environ.get(ALLOWED_ROOTS_ENV, "").split(":")
        if path.startswith("/") and "\n" not in path and "\r" not in path
    ]


def _confine_run_paths(template: str, root: str | None) -> str | None:
    """Confine run_case's template/root to the allowed OpenFOAM roots.

    Candidate and allowed paths are canonicalised with ``realpath -m`` inside
    WSL (so symlinks and ``..`` cannot escape), because the default root is
    only meaningful there.  Returns an error string, or None when every
    caller-provided path stays inside an allowed root.

    Not TOCTOU-free: the filesystem could change between this check and the
    case setup; the threat model is a confused agent, not a local racer.
    """
    extra = _extra_allowed_roots()
    kinds = (["root"] if root is not None else []) + ["template"]
    candidates = ([root] if root is not None else []) + [template]
    command = "realpath -m -- " + " ".join(
        [DEFAULT_ROOT_BASH] + [shlex.quote(path) for path in extra + candidates]
    )
    code, output, wsl_stderr = _run_parts(
        ["wsl.exe", "-e", "bash", "-lc", command], 30
    )
    if code is None:
        return f"could not resolve paths in WSL: {wsl_stderr}"
    if code != 0:
        return f"realpath failed: {_tail(_combine(output, wsl_stderr))}"
    resolved = [line.strip() for line in output.splitlines() if line.strip()]
    expected = 1 + len(extra) + len(candidates)
    if len(resolved) != expected:
        return (
            f"unexpected realpath output: got {len(resolved)} paths, "
            f"expected {expected}"
        )
    allowed = resolved[: 1 + len(extra)]
    allowed_text = ", ".join(allowed)
    for kind, resolved_path in zip(kinds, resolved[1 + len(extra) :]):
        if not _within_allowed(resolved_path, allowed):
            return (
                f"{kind} must stay inside an allowed OpenFOAM root "
                f"({allowed_text}); extend {ALLOWED_ROOTS_ENV} to add one "
                f"(got: {resolved_path})"
            )
    return None


def _build_case_command(
    template: str,
    name: str,
    n_correctors: int,
    dt: float | None,
    end_time: float | None,
    dx: float | None,
    solver: str,
    root: str | None,
) -> str:
    """Build the bash command line for make_of_case.sh.

    Every value that comes from the caller is shell-quoted; only the built-in
    default root is inserted as a raw bash expression (``"$HOME/..."``) so that
    it expands inside WSL. This keeps a hostile ``template``/``solver`` from
    escaping into the shell.
    """
    parts = [
        "bash",
        shlex.quote(_to_wsl(OF_SCRIPTS / "make_of_case.sh")),
        "--template",
        shlex.quote(template),
        "--name",
        shlex.quote(name),
        "--nc",
        str(int(n_correctors)),
        "--solver",
        shlex.quote(solver),
    ]
    if dt is not None:
        parts += ["--dt", repr(float(dt))]
    if end_time is not None:
        parts += ["--end", repr(float(end_time))]
    if dx is not None:
        parts += ["--dx", repr(float(dx))]
    parts += ["--root", shlex.quote(root) if root is not None else DEFAULT_ROOT_BASH]
    parts += ["--setup-only"]
    return " ".join(parts)


# Live launcher processes (case_dir -> Popen) for runs started by this
# server instance.  Background processes spawned from a `wsl.exe -e`
# invocation are killed when that invocation ends, so the run's wsl.exe must
# stay a live child of this (long-lived) server process; the handle also
# lets run_case report start_failed.  Runs started by a previous server
# instance simply have no entry here.
_RUNS: dict[str, subprocess.Popen] = {}

# OpenFOAM bashrc; a bash expression, expanded inside WSL.  make_of_case.sh
# honours FOAM_BASHRC; the launcher below does not (the MCP caller has no
# way to set it), so custom environments need the manual fallback.
_BASHRC_BASH = '"$HOME/OpenFOAM/OpenFOAM-14/etc/bashrc"'


def _launch_run(case_dir: str, solver: str) -> None:
    """Start foamRun for a prepared case as a live wsl.exe child of this
    process.  Writes the launcher PID, log.run and the foamRun exit code."""
    # The trailing brace group (not a simple command) keeps bash from
    # exec-optimising itself away: /proc/<launcher_pid>/cmdline must keep the
    # case path, because run_status' liveness check greps for it.
    launcher = (
        f"source {_BASHRC_BASH} >/dev/null; "
        f"cd {shlex.quote(case_dir)} && {{ echo $$ > .launcher_pid; "
        f"foamRun -solver {shlex.quote(solver)}"
        " > log.run 2>&1; echo $? > .exitcode; }"
    )
    try:
        _RUNS[case_dir] = subprocess.Popen(
            ["wsl.exe", "-e", "bash", "-lc", launcher],
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "WSL_UTF8": "1"},
        )
    except OSError:
        _RUNS.pop(case_dir, None)


def run_case(
    template: str,
    name: str,
    n_correctors: int,
    dt: float | None = None,
    end_time: float | None = None,
    dx: float | None = None,
    solver: str | None = None,
    root: str | None = None,
    timeout_s: int | None = None,
) -> tuple[str, bool]:
    if not template:
        return _error("template is required")
    if not template.startswith("/") or "\n" in template or "\r" in template:
        return _error("template must be an absolute WSL path without newlines", template=template)
    if template != template.strip():
        return _error(
            "template must not have leading/trailing whitespace", template=template
        )
    if not NAME_RE.match(name or ""):
        return _error("invalid name: use [A-Za-z0-9._-]+", name=name)
    if not isinstance(n_correctors, int) or n_correctors < 1:
        return _error("n_correctors must be an integer >= 1", n_correctors=n_correctors)
    for label, value in (("dt", dt), ("end_time", end_time), ("dx", dx)):
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            return _error(f"{label} must be a positive number", **{label: value})
    problem = _check_root(root)
    if problem:
        return _error(problem)
    solver = solver or DEFAULT_SOLVER
    if not SOLVER_RE.match(solver):
        return _error("invalid solver name", solver=solver)
    # Confinement comes after the cheap syntactic checks (including the
    # solver name): it costs one WSL round-trip (realpath), so invalid
    # requests fail fast without it.
    problem = _confine_run_paths(template, root)
    if problem:
        return _error(problem)
    # The foamRun itself runs in the background; this timeout only covers the
    # setup (copy + dictionary patching + launch), which takes seconds.
    timeout = 120
    if timeout_s is not None:
        try:
            timeout = max(1, min(int(timeout_s), MAX_TIMEOUT))
        except (TypeError, ValueError):
            return _error("timeout_s must be an integer", timeout_s=timeout_s)

    command = _build_case_command(
        template, name, n_correctors, dt, end_time, dx, solver, root
    )

    code, output, wsl_stderr = _run_parts(["wsl.exe", "-e", "bash", "-lc", command], timeout)
    if code is None:
        return _error("failed to start the case", detail=wsl_stderr)
    if code != 0:
        return _error(
            "make_of_case.sh failed", exit_code=code, output=_tail(_combine(output, wsl_stderr))
        )

    match = re.search(r"^case: (.+)$", output, re.MULTILINE)
    if not match:
        return _error(
            "could not determine the case directory",
            output=_tail(_combine(output, wsl_stderr)),
        )
    case_dir = match.group(1).strip()

    _launch_run(case_dir, solver)
    started = case_dir in _RUNS

    payload: dict = {
        "case_dir": case_dir,
        "log_path": f"{case_dir}/log.run",
        "status": "running" if started else "start_failed",
        "poll": (
            "call run_status with this case_dir until the state is terminal: "
            "'finished'/'failed'/'stale'"
        ),
        "config": {
            "template": template,
            "name": name,
            "n_correctors": n_correctors,
            "dt": dt,
            "end_time": end_time,
            "dx": dx,
            "solver": solver,
            "root": root or case_dir.rsplit("/", 1)[0],
        },
        "output": _tail(output),
    }
    return _as_json(payload)


_RUN_STATUS_SNIPPET = r"""
cd "$1" || { echo "NO_CASE $1" >&2; exit 1; }
[ -f mcp_run ] && cat mcp_run
echo "---STATE---"
if [ -f log.run ] && grep -q "FOAM FATAL" log.run; then
    echo "STATE=failed"
elif [ -f .exitcode ]; then
    echo "STATE=finished"
    echo "EXIT=$(cat .exitcode)"
elif [ -f log.run ]; then
    if [ -f .launcher_pid ]; then
        pid=$(cat .launcher_pid 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \
           && grep -qaF -- "$1" "/proc/$pid/cmdline" 2>/dev/null; then
            echo "STATE=running"
        else
            echo "STATE=stale"
        fi
    else
        echo "STATE=running"
        echo "PIDFILE=missing"
    fi
else
    echo "STATE=unknown"
fi
exit 0
"""


def run_status(
    case_dir: str, dt: float | None = None, dx: float | None = None
) -> tuple[str, bool]:
    problem = _check_path_arg("case_dir", case_dir)
    if problem:
        return _error(problem)
    if not case_dir:
        return _error("case_dir is required")

    code, output, stderr = _run_parts(
        ["wsl.exe", "-e", "bash", "-lc", _RUN_STATUS_SNIPPET, "_", case_dir], 60
    )
    if code is None:
        return _error("failed to poll the case", detail=stderr)
    if code != 0:
        return _error(
            "cannot read the case directory",
            case_dir=case_dir,
            output=_tail(_combine(output, stderr)),
        )

    meta: dict[str, float] = {}
    state = "unknown"
    exit_code = None
    pidfile_missing = False
    payload_note = None
    for line in output.splitlines():
        if line == "---STATE---":
            continue
        match = re.match(r"^(dt|dx)=([0-9.eE+-]+)$", line)
        if match:
            meta[match.group(1)] = float(match.group(2))
        elif line.startswith("STATE="):
            state = line[len("STATE=") :].strip()
        elif line == "PIDFILE=missing":
            pidfile_missing = True
        elif line.startswith("EXIT="):
            try:
                exit_code = int(line[len("EXIT=") :].strip())
            except ValueError:
                pass
    if state == "finished" and exit_code not in (None, 0):
        state = "failed"
    if state == "running":
        if pidfile_missing:
            payload_note = (
                "no .launcher_pid (case started by an older server); "
                "liveness unknown"
            )
    elif state == "stale":
        payload_note = (
            "the launcher process is dead and no .exitcode was written; "
            "the run died with the server or WSL (e.g. a restart). Rerun "
            "the case for a fresh comparison."
        )

    payload: dict = {
        "case_dir": case_dir,
        "log_path": f"{case_dir}/log.run",
        "state": state,
    }
    if payload_note:
        payload["note"] = payload_note
    if exit_code is not None:
        payload["exit_code"] = exit_code
    if state == "unknown":
        payload["note"] = "no log.run yet; the run may still be starting"
        return _as_json(payload)

    eff_dt = dt if dt is not None else meta.get("dt")
    eff_dx = dx if dx is not None else meta.get("dx")
    words = ["python3", _to_wsl(OF_SCRIPTS / "of_log_summary.py"), f"{case_dir}/log.run", "--json"]
    if eff_dt is not None and eff_dx is not None:
        words += ["--dt", repr(float(eff_dt)), "--dx", repr(float(eff_dx))]
    sum_code, sum_output, sum_stderr = _bash(words, [], 120)
    if sum_code == 0:
        try:
            payload["summary"] = _loads(sum_output)
        except json.JSONDecodeError:
            payload["summary_error"] = "summary was not valid JSON"
    else:
        payload["summary_error"] = _tail(_combine(sum_output, sum_stderr))
    return _as_json(payload)


def summarize_log(log: str, dt: float | None = None, dx: float | None = None) -> tuple[str, bool]:
    if not log:
        return _error("log is required")
    words = ["python3", _to_wsl(OF_SCRIPTS / "of_log_summary.py"), log, "--json"]
    if dt is not None:
        words += ["--dt", repr(float(dt))]
    if dx is not None:
        words += ["--dx", repr(float(dx))]
    code, output, wsl_stderr = _bash(words, [], 120)
    if code is None:
        return _error("failed to summarise the log", detail=wsl_stderr)
    if code != 0:
        return _error(
            "of_log_summary.py failed",
            exit_code=code,
            output=_tail(_combine(output, wsl_stderr)),
        )
    try:
        return _as_json(_loads(output))
    except json.JSONDecodeError:
        return _error("summary was not valid JSON", output=_tail(_combine(output, wsl_stderr)))


_LIST_SNIPPET = r"""
root="${1:-}"
if [ -z "$root" ]; then root="$HOME/OpenFOAM/gzh1057-14/run"; fi
filter="${2:-}"; limit="${3:-20}"
echo "ROOT=$root"
if [ ! -d "$root" ]; then echo "NO_ROOT $root" >&2; exit 1; fi
count=0
while IFS=$'\t' read -r name mtime; do
    case "$name" in *"$filter"*) ;; *) continue ;; esac
    if [ -f "$root/$name/log.run" ]; then has=yes; else has=no; fi
    printf '%s\t%s\t%s\n' "$name" "$mtime" "$has"
    count=$((count + 1))
    [ "$count" -ge "$limit" ] && break
done < <(find "$root" -maxdepth 1 -mindepth 1 -type d -printf '%f\t%T@\n' | sort -k2 -rn)
exit 0
"""


def list_cases(
    root: str | None = None, name_filter: str | None = None, limit: int | None = None
) -> tuple[str, bool]:
    problem = _check_root(root)
    if problem:
        return _error(problem)
    try:
        count = max(1, min(int(limit), 200)) if limit is not None else 20
    except (TypeError, ValueError):
        return _error("limit must be an integer", limit=limit)

    # Empty argument -> the snippet expands $HOME itself (argv is never expanded).
    root_arg = root if root is not None else ""
    code, output, wsl_stderr = _run_parts(
        [
            "wsl.exe",
            "-e",
            "bash",
            "-lc",
            _LIST_SNIPPET,
            "_",
            root_arg,
            name_filter or "",
            str(count),
        ],
        120,
    )
    if code is None:
        return _error("failed to list cases", detail=wsl_stderr)
    if code != 0:
        return _error(
            "failed to list cases",
            exit_code=code,
            output=_tail(_combine(output, wsl_stderr)),
        )

    resolved_root = None
    cases = []
    for line in output.splitlines():
        if line.startswith("ROOT="):
            resolved_root = line[len("ROOT=") :].strip()
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        dir_name, mtime, has_log = parts
        entry: dict = {"dir_name": dir_name, "mtime": float(mtime), "has_log": has_log == "yes"}
        match = CASE_RE.match(dir_name)
        if match:
            entry["name"] = match.group("name")
            entry["n_correctors"] = int(match.group("nc"))
            entry["dt"] = float(match.group("dt")) if match.group("dt") else None
            entry["end_time"] = float(match.group("t")) if match.group("t") else None
        cases.append(entry)
    return _as_json(
        {"root": resolved_root or root or "~", "n_cases": len(cases), "cases": cases}
    )


# ---------------------------------------------------------------------------
# Source lookup (find / read) and field extraction
# ---------------------------------------------------------------------------

DEFAULT_OF14_BASH = '"$HOME/OpenFOAM/OpenFOAM-14"'

ENV_PROPS = {
    "of14": {
        "type": "string",
        "description": "OpenFOAM-14 tree in WSL (default ~/OpenFOAM/OpenFOAM-14).",
    },
    "root": {
        "type": "string",
        "description": "Case root in WSL (default ~/OpenFOAM/gzh1057-14/run).",
    },
}

FIND_SOURCE_DESCRIPTION = (
    "Search the OpenFOAM-14 source tree in WSL with grep -rn (scope: src, "
    "applications, tutorials or all) and return file:line matches, optionally "
    "with context. Use this to check the reference implementation behind a "
    "cross-check difference. OF-14 layout note: there is no pEqn.H/UEqn.H for "
    "the modern solvers - the incompressible pressure correction is "
    "applications/modules/incompressibleFluid/correctPressure.C, the momentum "
    "predictor is momentumPredictor.C in the same directory, and fvMatrix "
    "internals are under src/finiteVolume/fvMatrices/fvMatrix/."
)

FIND_SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "grep pattern (extended regex by default)."},
        "scope": {
            "type": "string",
            "enum": ["all", "src", "applications", "tutorials"],
            "description": "Subtree to search (default all = src + applications).",
        },
        "limit": {"type": "integer", "minimum": 1, "description": "Max output lines (default 40)."},
        "context": {
            "type": "integer",
            "minimum": 0,
            "description": "Lines of context around each match (default 0).",
        },
        "fixed": {
            "type": "boolean",
            "description": "Treat the pattern as a literal string (grep -F).",
        },
        **ENV_PROPS,
    },
    "required": ["pattern"],
    "additionalProperties": False,
}

READ_SOURCE_DESCRIPTION = (
    "Read a numbered snippet of a text file in WSL - an OpenFOAM-14 source "
    "file (relative to ~/OpenFOAM/OpenFOAM-14, or absolute under it) or a case "
    "file under the case root (e.g. a patched fvSolution). Use it after "
    "find_source to read the code behind a difference."
)

READ_SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Relative to OpenFOAM-14, or absolute inside OpenFOAM-14 / the case root.",
        },
        "start": {"type": "integer", "minimum": 1, "description": "First line (default 1)."},
        "lines": {
            "type": "integer",
            "minimum": 1,
            "description": "Number of lines to read (default 120, max 1000).",
        },
        **ENV_PROPS,
    },
    "required": ["path"],
    "additionalProperties": False,
}

EXTRACT_FIELDS_DESCRIPTION = (
    "Read fields from a finished OpenFOAM case time directory (ascii) and map "
    "them onto this project's layout when nx/ny are given: cell fields as "
    "grid[j][i] (CartesianMesh::cellIndex = j*nx + i; the cell order matches "
    "blockMesh for the cases used here - verified against p(0,0)), phi as "
    "x_faces[j][i] / y_faces[j][i] in this project's sign convention (positive "
    "along +x/+y; owner/neighbour are used, so the OpenFOAM orientation does "
    "not matter), plus per-patch boundary values with their owner cells, a "
    "West/East/South/North side guess and the sign that converts them to this "
    "project's face storage. nx and ny must equal the mesh's own cell counts "
    "(nCells): a mismatch is rejected with 'nx*ny does not match the mesh cell "
    "count' instead of mapping onto a wrong grid. time='latest' (default) "
    "takes the largest numeric time directory."
)

EXTRACT_FIELDS_SCHEMA = {
    "type": "object",
    "properties": {
        "case_dir": {"type": "string", "description": "Case directory in WSL (absolute)."},
        "time": {
            "type": "string",
            "description": "Time directory name, or 'latest' (default).",
        },
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Field names to read (default [p, U, phi]).",
        },
        "nx": {"type": "integer", "minimum": 1, "description": "Cells along x (enables the mapping)."},
        "ny": {"type": "integer", "minimum": 1, "description": "Cells along y (enables the mapping)."},
        "max_values": {
            "type": "integer",
            "minimum": 1,
            "description": "Max values returned per flat list (default 24).",
        },
        "include_grid": {
            "type": "boolean",
            "description": "Include the full (i,j) grids (default true).",
        },
    },
    "required": ["case_dir"],
    "additionalProperties": False,
}

FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
TIME_NAME_RE = re.compile(r"^[0-9][0-9.eE+-]*$")
EXTRACT_FIELDS_MAX_CHARS = 30000


def _check_path_arg(label: str, value: str | None) -> str | None:
    """Returns an error string, or None when the WSL path looks acceptable."""
    if value is None:
        return None
    if not value.startswith("/"):
        return f"{label} must be an absolute WSL path, got: {value}"
    if "\n" in value or "\r" in value:
        return f"{label} must not contain newlines"
    return None


def _env_prefix(of14: str | None, root: str | None) -> str:
    """Bash lines defining OF14 and ROOT (defaults expand inside WSL)."""
    of14_expr = shlex.quote(of14) if of14 is not None else DEFAULT_OF14_BASH
    root_expr = shlex.quote(root) if root is not None else DEFAULT_ROOT_BASH
    return f"OF14={of14_expr}\nROOT={root_expr}\n"


_FIND_SNIPPET = r"""
cd "$OF14" || { echo "NO_OF14 $OF14" >&2; exit 1; }
pat="$1"; scope="$2"; limit="$3"; ctx="$4"; fixed="$5"
case "$scope" in
    src)          dirs="src" ;;
    applications) dirs="applications" ;;
    tutorials)    dirs="tutorials" ;;
    *)            dirs="src applications" ;;
esac
args=(-rnI -m "$limit")
[ "$fixed" = "yes" ] && args+=(-F)
[ "$ctx" != "0" ] && args+=(-C "$ctx")
# shellcheck disable=SC2086  # $dirs is intentionally word-split
grep "${args[@]}" -e "$pat" $dirs 2>/dev/null | head -n "$limit"
exit 0
"""

GREP_MATCH_RE = re.compile(
    r"^(?P<file>[^:]+)(?P<sep>[:\-])(?P<line>\d+)(?P=sep)(?P<text>.*)$"
)


def find_source(
    pattern: str,
    scope: str = "all",
    limit: int | None = None,
    context: int | None = None,
    fixed: bool | None = None,
    of14: str | None = None,
    root: str | None = None,
) -> tuple[str, bool]:
    if not pattern:
        return _error("pattern is required")
    if "\n" in pattern or "\r" in pattern:
        return _error("pattern must not contain newlines", pattern=pattern)
    if scope not in ("all", "src", "applications", "tutorials"):
        return _error("invalid scope", scope=scope)
    for label, value in (("of14", of14), ("root", root)):
        problem = _check_path_arg(label, value)
        if problem:
            return _error(problem)
    try:
        max_lines = max(1, min(int(limit), 400)) if limit is not None else 40
        ctx = max(0, min(int(context), 10)) if context is not None else 0
    except (TypeError, ValueError):
        return _error("limit/context must be integers")

    command = _env_prefix(of14, root) + _FIND_SNIPPET
    code, output, stderr = _run_parts(
        [
            "wsl.exe",
            "-e",
            "bash",
            "-lc",
            command,
            "_",
            pattern,
            scope,
            str(max_lines),
            str(ctx),
            "yes" if fixed else "no",
        ],
        120,
    )
    if code is None:
        return _error("failed to search the OpenFOAM tree", detail=stderr)
    if code != 0:
        return _error(
            "find_source failed",
            exit_code=code,
            output=_tail(_combine(output, stderr)),
        )

    matches = []
    for line in output.splitlines():
        match = GREP_MATCH_RE.match(line)
        if not match:
            continue
        matches.append(
            {
                "file": match.group("file"),
                "line": int(match.group("line")),
                "match": match.group("sep") == ":",
                "text": match.group("text"),
            }
        )
    payload = {
        "pattern": pattern,
        "scope": scope,
        "of14": of14 or "~/OpenFOAM/OpenFOAM-14",
        "n_matches": sum(1 for m in matches if m["match"]),
        "matches": matches,
        "raw": _tail(output),
    }
    if not matches:
        payload["note"] = "no matches (check the pattern, or the OF14 tree path)"
    return _as_json(payload)


_READ_SNIPPET = r"""
p="$1"; start="$2"; end="$3"
case "$p" in
    /*) f="$p" ;;
    *)  if [ -f "$OF14/$p" ]; then f="$OF14/$p"; else f="$ROOT/$p"; fi ;;
esac
case "$f" in
    "$OF14"/*|"$ROOT"/*) ;;
    *) echo "path must be inside OF14 or ROOT: $f" >&2; exit 2 ;;
esac
[ -f "$f" ] || { echo "no such file: $f" >&2; exit 1; }
total=$(wc -l < "$f")
awk -v s="$start" -v e="$end" 'NR>=s && NR<=e {printf "%6d  %s\n", NR, $0}' "$f"
echo "---"
echo "__FILE__=$f"
echo "__TOTAL__=$total"
exit 0
"""


def read_source(
    path: str,
    start: int | None = None,
    lines: int | None = None,
    of14: str | None = None,
    root: str | None = None,
) -> tuple[str, bool]:
    if not path:
        return _error("path is required")
    if "\n" in path or "\r" in path:
        return _error("path must not contain newlines", path=path)
    for label, value in (("of14", of14), ("root", root)):
        problem = _check_path_arg(label, value)
        if problem:
            return _error(problem)
    try:
        first = max(1, int(start)) if start is not None else 1
        count = max(1, min(int(lines), 1000)) if lines is not None else 120
    except (TypeError, ValueError):
        return _error("start/lines must be integers")
    last = first + count - 1

    command = _env_prefix(of14, root) + _READ_SNIPPET
    code, output, stderr = _run_parts(
        [
            "wsl.exe",
            "-e",
            "bash",
            "-lc",
            command,
            "_",
            path,
            str(first),
            str(last),
        ],
        120,
    )
    if code is None:
        return _error("failed to read the file", detail=stderr)
    if code != 0:
        return _error(
            "read_source failed",
            exit_code=code,
            path=path,
            output=_tail(_combine(output, stderr)),
        )

    resolved = None
    total = None
    body_lines = []
    for line in output.splitlines():
        if line.startswith("__FILE__="):
            resolved = line[len("__FILE__=") :]
        elif line.startswith("__TOTAL__="):
            total = int(line[len("__TOTAL__=") :] or 0)
        elif line == "---":
            continue
        else:
            body_lines.append(line)
    return _as_json(
        {
            "file": resolved or path,
            "start": first,
            "end": min(last, total) if total else last,
            "total_lines": total,
            "text": "\n".join(body_lines),
        }
    )


def _shrink_json(payload: dict, budget: int) -> str:
    """Serialize to valid JSON within ``budget`` chars, dropping arrays if needed.

    A hard ``_tail`` cut would leave invalid JSON, so the payload is degraded
    step by step instead: pretty -> compact -> grid arrays dropped -> patch
    arrays dropped -> first_values dropped. The tool result therefore always
    parses, and ``truncated_payload`` says what was removed.
    """

    def dumps(pretty: bool) -> str:
        if pretty:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    text = dumps(True)
    if len(text) <= budget:
        return text
    text = dumps(False)
    if len(text) <= budget:
        return text

    removed: list[str] = []

    def pop_all(keys: tuple[str, ...]) -> bool:
        changed = False
        for key in keys:
            for entry in payload["fields"].values():
                if isinstance(entry, dict) and key in entry:
                    entry.pop(key)
                    changed = True
        if changed:
            removed.extend(keys)
        return changed

    # Redundant arrays first, then the grids themselves.
    if pop_all(("first_values",)):
        text = dumps(False)
    if len(text) > budget and pop_all(("grid", "x_faces", "y_faces")):
        text = dumps(False)

    if len(text) > budget:
        changed = False
        for entry in payload["fields"].values():
            if not isinstance(entry, dict):
                continue
            for patch in entry.get("boundary_patches") or []:
                for key in ("values", "values_ours", "owner_cells"):
                    if key in patch:
                        patch.pop(key)
                        changed = True
        if changed:
            removed.append("boundary_patch_arrays")
        text = dumps(False)

    if len(text) > budget:
        for entry in payload["fields"].values():
            if isinstance(entry, dict) and "first_values" in entry:
                entry.pop("first_values")
        removed.append("first_values")

    if removed:
        payload["truncated_payload"] = {
            "reason": f"result exceeded {budget} chars; removed: {', '.join(removed)}",
            "hint": "request fewer fields, set include_grid=false, or lower max_values",
        }
    return dumps(False)


def extract_fields(
    case_dir: str,
    time: str | None = None,
    fields: list[str] | None = None,
    nx: int | None = None,
    ny: int | None = None,
    max_values: int | None = None,
    include_grid: bool | None = None,
) -> tuple[str, bool]:
    problem = _check_path_arg("case_dir", case_dir)
    if problem:
        return _error(problem)
    if not case_dir:
        return _error("case_dir is required")
    if (nx is None) != (ny is None):
        return _error("nx and ny must be given together")
    if nx is not None and (int(nx) < 1 or int(ny) < 1):
        return _error("nx/ny must be positive")
    names = fields or ["p", "U", "phi"]
    for name in names:
        if not FIELD_NAME_RE.match(name):
            return _error("invalid field name", field=name)
    try:
        max_vals = max(1, min(int(max_values), 4096)) if max_values is not None else 24
    except (TypeError, ValueError):
        return _error("max_values must be an integer")
    want_grid = True if include_grid is None else bool(include_grid)
    # ---- resolve the time directory ------------------------------------
    code, output, stderr = _bash(
        ["find", case_dir, "-maxdepth", "1", "-mindepth", "1", "-type", "d", "-printf", "%f\n"],
        [],
        60,
    )
    if code is None:
        return _error("failed to list the case directory", detail=stderr)
    if code != 0:
        return _error(
            "cannot list case_dir", case_dir=case_dir, output=_tail(_combine(output, stderr))
        )
    times = [t for t in output.splitlines() if TIME_NAME_RE.match(t)]
    times.sort(key=float)
    if time in (None, "latest"):
        if not times:
            return _error("no numeric time directory found", case_dir=case_dir)
        chosen = times[-1]
    else:
        if not TIME_NAME_RE.match(time):
            return _error("time must be a number or 'latest'", time=time)
        if time not in times:
            return _error(
                "time directory not found", time=time, available=times[-10:]
            )
        chosen = time

    def read_file(rel: str) -> str | None:
        code, out, err = _bash(["cat", f"{case_dir}/{rel}"], [], 120)
        if code != 0:
            return None
        return out

    payload: dict = {
        "case_dir": case_dir,
        "time": chosen,
        "available_times": times,
        "nx": nx,
        "ny": ny,
        "fields": {},
        "warnings": [],
    }

    # ---- writeFormat: the reader is ascii-only --------------------------
    control_text = read_file("system/controlDict")
    if control_text is None:
        payload["warnings"].append(
            "system/controlDict not readable; writeFormat unknown"
        )
    elif of_fields.parse_write_format(control_text) == "binary":
        return _error(
            "writeFormat binary is not supported by the ascii reader; set "
            "writeFormat ascii in system/controlDict",
            case_dir=case_dir,
            time=chosen,
        )

    # ---- polyMesh (only needed for the mapping) ------------------------
    owners: list[int] = []
    neighbours: list[int] = []
    n_internal = None
    mesh_patches: list[dict] = []
    if nx is not None:
        owner_text = read_file("constant/polyMesh/owner")
        neighbour_text = read_file("constant/polyMesh/neighbour")
        boundary_text = read_file("constant/polyMesh/boundary")
        if owner_text is None or neighbour_text is None or boundary_text is None:
            payload["warnings"].append(
                "constant/polyMesh/{owner,neighbour,boundary} not readable; "
                "mapping disabled"
            )
        else:
            mesh = of_fields.parse_poly_mesh(
                owner_text, neighbour_text, boundary_text
            )
            payload["warnings"].extend(mesh["warnings"])
            owners = mesh["owners"]
            neighbours = mesh["neighbours"]
            n_internal = mesh["n_internal"]
            mesh_patches = mesh["patches"]
            expected_cells = int(nx) * int(ny)
            if mesh["n_cells"] is not None and expected_cells != mesh["n_cells"]:
                return _error(
                    "nx*ny does not match the mesh cell count",
                    nx=nx,
                    ny=ny,
                    nx_times_ny=expected_cells,
                    n_cells=mesh["n_cells"],
                    hint=(
                        "pass the mesh's own (i,j) cell counts; see "
                        "constant/polyMesh/owner"
                    ),
                )

    # ---- read and map each field ---------------------------------------
    for name in names:
        raw = read_file(f"{chosen}/{name}")
        if raw is None:
            payload["warnings"].append(f"field {name} not found in time {chosen}")
            continue
        if of_fields.looks_binary(raw):
            payload["warnings"].append(
                f"field {name} appears binary; set writeFormat ascii"
            )
            payload["fields"][name] = {"class": None, "internal": None, "n": None}
            continue
        text = of_fields.strip_foam_comments(raw)
        class_match = re.search(r"\bclass\s+(\w+)\s*;", text)
        field_class = class_match.group(1) if class_match else None
        is_surface = field_class == "surfaceScalarField" or name == "phi"
        kind, values, count = of_fields.parse_foam_list(text, "internalField")
        entry: dict = {"class": field_class, "internal": kind, "n": count}
        if values is None:
            payload["warnings"].append(f"internalField of {name} not parsed")
            payload["fields"][name] = entry
            continue
        if count == 1 and kind == "uniform":
            entry["uniform"] = values
            payload["fields"][name] = entry
            continue

        flat = values
        is_vector = bool(flat) and isinstance(flat[0], list)
        entry["is_vector"] = is_vector
        entry["min"], entry["max"] = of_fields.min_max(flat, is_vector)

        has_grid = False
        if nx is not None and not is_surface:
            if len(flat) == nx * ny:
                entry["grid_note"] = (
                    "grid[j][i] = cell (i, j); matches "
                    "CartesianMesh::cellIndex = j*nx + i"
                )
                if want_grid:
                    entry.update(of_fields.build_cell_grids(flat, nx, ny))
                    has_grid = True
            else:
                payload["warnings"].append(
                    f"{name}: {len(flat)} internal values != nx*ny ({nx * ny}) "
                    "but the field is not a surface field; returning the flat list only"
                )
        if not has_grid:
            entry["first_values"] = flat[:max_vals]
            entry["truncated"] = len(flat) > max_vals
        payload["fields"][name] = entry

    # ---- phi: internal face mapping + boundary patches ------------------
    if (
        "phi" in payload["fields"]
        and nx is not None
        and want_grid
        and owners
        and neighbours
    ):
        phi_text = read_file(f"{chosen}/phi") or ""
        entry = payload["fields"]["phi"]
        if not of_fields.looks_binary(phi_text):
            # a binary phi was already warned about in the field loop above
            _, phi_values, _ = of_fields.parse_foam_list(
                of_fields.strip_foam_comments(phi_text), "internalField"
            )
            if phi_values is None:
                payload["warnings"].append("phi internalField not parsed")
            elif n_internal is None or len(phi_values) < n_internal:
                payload["warnings"].append(
                    f"phi has {len(phi_values)} internal values but the mesh expects "
                    f"{n_internal}; mapping skipped"
                )
            else:
                x_faces, y_faces, unmapped = of_fields.map_internal_fluxes(
                    phi_values, owners, neighbours, n_internal, nx, ny
                )
                entry["x_faces"] = x_faces
                entry["y_faces"] = y_faces
                entry["face_note"] = (
                    "x_faces[j][i] = FaceFluxField::x(i, j), y_faces[j][i] = "
                    "FaceFluxField::y(i, j); signs converted from OpenFOAM's "
                    "owner->neighbour convention to +x/+y"
                )
                if unmapped:
                    payload["warnings"].append(
                        f"{unmapped} internal faces are not axis-aligned neighbours "
                        "(unstructured mesh?) and were left unset"
                    )

                # boundary patches: owner cells + values + side guess
                patch_values = of_fields.parse_boundary_patch_values(phi_text)
                entry["boundary_patches"] = of_fields.build_boundary_patches(
                    mesh_patches, patch_values, owners, nx, ny, max_vals
                )

    return _shrink_json(payload, EXTRACT_FIELDS_MAX_CHARS), True


# ---------------------------------------------------------------------------
# Minimal MCP plumbing (newline-delimited JSON-RPC 2.0 over stdio)
# ---------------------------------------------------------------------------

TOOLS = [
    ("run_case", RUN_CASE_DESCRIPTION, RUN_CASE_SCHEMA, run_case),
    ("run_status", RUN_STATUS_DESCRIPTION, RUN_STATUS_SCHEMA, run_status),
    ("summarize_log", SUMMARIZE_LOG_DESCRIPTION, SUMMARIZE_LOG_SCHEMA, summarize_log),
    ("list_cases", LIST_CASES_DESCRIPTION, LIST_CASES_SCHEMA, list_cases),
    ("extract_fields", EXTRACT_FIELDS_DESCRIPTION, EXTRACT_FIELDS_SCHEMA, extract_fields),
    ("find_source", FIND_SOURCE_DESCRIPTION, FIND_SOURCE_SCHEMA, find_source),
    ("read_source", READ_SOURCE_DESCRIPTION, READ_SOURCE_SCHEMA, read_source),
]


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _reply(request_id, result=None, error=None) -> None:
    message = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    _send(message)


def _handle(message: dict) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if request_id is None:
        return  # notification (e.g. notifications/initialized)

    if method == "initialize":
        _reply(
            request_id,
            {
                "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    elif method == "ping":
        _reply(request_id, {})
    elif method == "tools/list":
        _reply(
            request_id,
            {
                "tools": [
                    {"name": name, "description": description, "inputSchema": schema}
                    for name, description, schema, _ in TOOLS
                ]
            },
        )
    elif method == "tools/call":
        name = params.get("name")
        handler = next((fn for tool, _, _, fn in TOOLS if tool == name), None)
        if handler is None:
            _reply(request_id, error={"code": -32602, "message": f"unknown tool: {name}"})
            return
        arguments = params.get("arguments") or {}
        try:
            text, ok = handler(**arguments)
        except TypeError as exc:
            text, ok = _error(f"invalid arguments: {exc}")
        except Exception as exc:  # never take the server down
            text, ok = _error(f"{name} crashed: {exc!r}")
        _reply(request_id, {"content": [{"type": "text", "text": text}], "isError": not ok})
    else:
        _reply(request_id, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> int:
    try:
        sys.stdout.reconfigure(newline="\n")
    except (AttributeError, ValueError):
        pass
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            _handle(message)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
