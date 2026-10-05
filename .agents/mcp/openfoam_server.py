#!/usr/bin/env python3
"""Stdio MCP server exposing OpenFOAM comparison cases (WSL) as tools.

Speaks the same minimal subset of the Model Context Protocol as
``build_and_test_server.py`` (newline-delimited JSON-RPC 2.0 over
stdin/stdout) and uses only the Python standard library.

Shared by Codex (registered in ``.codex/config.toml``) and opencode (reached
through the ``mcp`` block in ``opencode.json``).  Because MCP servers run
outside the agent sandbox, this removes the per-run escalation that shelling
into WSL from an agent requires, and gives both agents one structured
interface instead of hand-rolled commands.

The server does **not** re-implement the workflow: it drives the scripts of
the ``openfoam-crosscheck`` skill, which stay the single source of truth:

* ``scripts/make_of_case.sh``   - build a fresh, timestamped case dir and run it
* ``scripts/of_log_summary.py`` - parse a solver log (``--json`` for machines)

Scope is deliberately narrow: the only mutation is creating a new case
directory and running ``foamRun`` in it.  There is no tool for arbitrary
commands and nothing deletes or overwrites existing cases.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

MAX_OUTPUT_CHARS = 8000
SERVER_NAME = "openfoam"
SERVER_VERSION = "1.0.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

RUN_TIMEOUT = 900
MAX_TIMEOUT = 3600

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILL_SCRIPTS = REPO_ROOT / ".agents" / "skills" / "openfoam-crosscheck" / "scripts"

# Bash expression, not a literal path: it must expand inside WSL.
DEFAULT_ROOT_BASH = '"$HOME/OpenFOAM/gzh1057-14/run"'
DEFAULT_SOLVER = "incompressibleFluid"

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
    "dictionaries (endTime/deltaT/writeInterval, nCorrectors), run foamRun and "
    "return a JSON summary (case_dir, log_path, per-step residuals/Courant and "
    "the last step's fields). Each configuration gets its own timestamped "
    "directory with the configuration baked into the name; nothing is "
    "overwritten. Use this instead of shelling into WSL by hand."
)

RUN_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "template": {
            "type": "string",
            "description": "Existing case directory to copy (WSL path).",
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
            "description": "Case root (WSL path; default ~/OpenFOAM/gzh1057-14/run).",
        },
        "timeout_s": {"type": "integer", "description": "Wall-clock timeout (default 900)."},
    },
    "required": ["template", "name", "n_correctors"],
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
        shlex.quote(_to_wsl(SKILL_SCRIPTS / "make_of_case.sh")),
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
    return " ".join(parts)


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
    timeout = RUN_TIMEOUT
    if timeout_s is not None:
        try:
            timeout = max(1, min(int(timeout_s), MAX_TIMEOUT))
        except (TypeError, ValueError):
            return _error("timeout_s must be an integer", timeout_s=timeout_s)

    summary_script = _to_wsl(SKILL_SCRIPTS / "of_log_summary.py")

    command = _build_case_command(
        template, name, n_correctors, dt, end_time, dx, solver, root
    )

    code, output, wsl_stderr = _run_parts(["wsl.exe", "-e", "bash", "-lc", command], timeout)
    if code is None:
        return _error("failed to run the case", detail=wsl_stderr)
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

    summary_words = ["python3", summary_script, f"{case_dir}/log.run", "--json"]
    if dt is not None and dx is not None:
        summary_words += ["--dt", repr(float(dt)), "--dx", repr(float(dx))]
    summary_code, summary_output, summary_stderr = _bash(summary_words, [], 120)

    payload: dict = {
        "case_dir": case_dir,
        "log_path": f"{case_dir}/log.run",
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
    }
    if summary_code == 0:
        try:
            payload["summary"] = _loads(summary_output)
        except json.JSONDecodeError:
            payload["summary_error"] = "summary was not valid JSON"
            payload["summary_raw"] = _tail(_combine(summary_output, summary_stderr))
    else:
        payload["summary_error"] = _tail(_combine(summary_output, summary_stderr))
    # stdout only: on success the WSL stderr banner is noise, on failure the
    # error paths above already include it.
    payload["output"] = _tail(output)
    return _as_json(payload)


def summarize_log(log: str, dt: float | None = None, dx: float | None = None) -> tuple[str, bool]:
    if not log:
        return _error("log is required")
    words = ["python3", _to_wsl(SKILL_SCRIPTS / "of_log_summary.py"), log, "--json"]
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
# Minimal MCP plumbing (newline-delimited JSON-RPC 2.0 over stdio)
# ---------------------------------------------------------------------------

TOOLS = [
    ("run_case", RUN_CASE_DESCRIPTION, RUN_CASE_SCHEMA, run_case),
    ("summarize_log", SUMMARIZE_LOG_DESCRIPTION, SUMMARIZE_LOG_SCHEMA, summarize_log),
    ("list_cases", LIST_CASES_DESCRIPTION, LIST_CASES_SCHEMA, list_cases),
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
