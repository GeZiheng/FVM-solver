#!/usr/bin/env python3
"""Stdio MCP server exposing this project's build-and-test workflow.

Speaks just enough of the Model Context Protocol (newline-delimited
JSON-RPC 2.0 over stdin/stdout) to expose two tools, ``build_and_test``
and ``test_status``. It uses only the Python standard library, so it
runs without installing an MCP SDK or reaching the network.

Shared by Codex (registered in ``.codex/config.toml``) and opencode
(reached through the ``mcp`` block in ``opencode.json``), so both agents
drive the same implementation. The tool configures with CMake + Ninja into
``build/<config>-agent``, builds ``fvm_tests``, then optionally runs ctest.
With ``python=true`` it configures ``-DFVM_BUILD_PYTHON=ON`` into
``build/<config>-agent-py`` instead, which adds the ``pyfvm`` extension
target and the ``pyfvm_tests`` ctest entry. ``run`` selects the ctest
layer (none / unit / application / all); the slow layers default to a
background ctest polled through ``test_status``.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time

MAX_OUTPUT_CHARS = 8000
SERVER_NAME = "build-and-test"
SERVER_VERSION = "1.4.0"
# Used only when the client does not request a version; otherwise the
# requested version is echoed back so the client keeps the negotiation.
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

CONFIGURE_TIMEOUT = 1800
BUILD_TIMEOUT = 1800
RUN_TIMEOUT = 1800
ENV_TIMEOUT = 300
ASYNC_TEST_TIMEOUT = 3600

RUN_MODES = ("none", "unit", "application", "all")

# In-memory handle for the background ctest started with wait=false; see
# _start_async_ctest()/test_status(). A marker file next to the log covers
# the "server died mid-run" case (test_status then reports stale).
_ACTIVE_CTEST: dict | None = None

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Used to reproduce the Visual Studio developer environment; see
# _msvc_environment(). vswhere.exe sits under a Program Files root, which the
# agent may filter out of the environment, so resolve it at runtime (see
# _find_vswhere) rather than hard-coding a drive/path.
VSWHERE_SUBPATH = (
    pathlib.Path("Microsoft Visual Studio") / "Installer" / "vswhere.exe"
)
VS_CPP_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

TOOL_DESCRIPTION = (
    "Configure (cmake -G Ninja) and build this FVM-solver project in Debug or "
    "Release mode (default Release), building fvm_tests (default) and, with "
    "python=true, the pyfvm extension, then optionally run ctest (which "
    "includes the pyfvm_tests Python binding tests when python=true). Use "
    "this instead of invoking cmake/ctest/ninja directly for project builds. "
    "run selects the ctest layer: unit (fvm_unit_tests + pyfvm_tests, "
    "excludes the application label, fast), application (only the labeled "
    "application tests; requires python=true), all (everything; the old "
    "value 'tests' is accepted as an alias). "
    "application and all exceed the client's ~300s tool-call timeout, so "
    "they default to wait=false: ctest starts in the background and the call "
    "returns immediately; poll the separate test_status tool until the run "
    "reaches a terminal state. Pass wait=true to force synchronous "
    "execution (the client may then time out even though ctest completes). "
    "Returns structured JSON. Failures include first_error, tail and "
    "log_path, and the complete untruncated transcript is written to "
    "build_and_test.log in the build directory. "
    "When the user directly requests a build or test, confirm the options "
    "(config, target, run, python) with the user first unless they already "
    "stated them. "
    "When building as part of a code modification workflow, proceed with the "
    "defaults (config=Release, target=all, run=unit, python=false). "
    "Success reports are compact by default (quiet=true). Failures return "
    "first_error, a bounded tail and log_path; quiet=false keeps a bounded "
    "sectioned transcript in report. The only untruncated transcript is "
    "build_and_test.log in the build directory."
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "config": {
            "type": "string",
            "enum": ["Debug", "Release"],
            "default": "Release",
            "description": "CMake build type: Debug or Release (default Release).",
        },
        "target": {
            "type": "string",
            "enum": ["all", "fvm_tests", "pyfvm"],
            "default": "all",
            "description": (
                "Which target(s) to build: all (default), fvm_tests, or pyfvm "
                "(pyfvm requires python=true)."
            ),
        },
        "run": {
            "type": "string",
            "enum": ["none", "unit", "application", "all"],
            "default": "unit",
            "description": (
                "What to run after a successful build: none; unit (ctest "
                "-LE application: fvm_unit_tests + pyfvm_tests, fast); "
                "application (ctest -L application: the labeled application "
                "tests, requires python=true); all (every registered test). "
                "application/all default to asynchronous execution with "
                "wait=false because they exceed the client tool-call "
                "timeout; poll test_status for the result."
            ),
        },
        "wait": {
            "type": "boolean",
            "description": (
                "Wait for ctest to finish inside the call (true) or start it "
                "in the background and return immediately (false). Defaults "
                "to the run mode: synchronous for none/unit, asynchronous "
                "for application/all. Only meaningful with run != none."
            ),
        },
        "python": {
            "type": "boolean",
            "default": False,
            "description": (
                "Build with -DFVM_BUILD_PYTHON=ON (pybind11 bindings) into "
                "build/<config>-agent-py, separate from the default build. "
                "ctest then also runs the Python binding tests."
            ),
        },
        "quiet": {
            "type": "boolean",
            "default": True,
            "description": (
                "Compact success report (default true): build/test summary "
                "only. Failures include first_error/tail/log_path; "
                "quiet=false keeps a bounded sectioned transcript in report "
                "(the untruncated transcript is only in build_and_test.log)."
            ),
        },
    },
    "required": [],
    "additionalProperties": False,
}

TEST_STATUS_DESCRIPTION = (
    "Poll the background ctest run started by build_and_test with "
    "wait=false (the default for run=application/all). Returns "
    "state=running with an elapsed time and a bounded log tail, or the "
    "terminal result parsed exactly like the synchronous path "
    "(ok/tests/summary/failed/log_path). A run older than "
    f"{ASYNC_TEST_TIMEOUT}s is killed and reported as failed. state=stale "
    "means the marker file survives but the server process holding the "
    "handle is gone (e.g. the session restarted mid-run): rerun the tests."
)

TEST_STATUS_SCHEMA = {
    "type": "object",
    "properties": {},
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


def _first_error(output: str) -> str | None:
    """Return the first material error line, by diagnostic category."""
    categories = (
        "CMake Error",
        "FAILED:",
        re.compile(r"error C\d+"),
        "error:",
        "The following tests FAILED:",
    )
    lines = output.splitlines()
    for category in categories:
        for line in lines:
            if isinstance(category, str):
                if category == "FAILED:":
                    matched = line.startswith(category)
                else:
                    matched = category in line
                if matched:
                    return line.strip()[:MAX_OUTPUT_CHARS]
            elif category.search(line):
                return line.strip()[:MAX_OUTPUT_CHARS]
    return None


def _parse_ctest(output: str) -> dict:
    """Parse ctest's summary and failed-test names from its output."""
    result: dict = {"summary": None, "total": None, "failed": []}
    summary = re.search(
        r"(\d+)% tests passed, (\d+) tests? failed out of (\d+)", output
    )
    if summary:
        result["summary"] = summary.group(0)
        result["total"] = int(summary.group(3))

    lines = output.splitlines()
    try:
        failed_header = lines.index("The following tests FAILED:")
    except ValueError:
        failed_header = None
    if failed_header is not None:
        for line in lines[failed_header + 1 :]:
            match = re.match(r"^\s*\d+\s+-\s+(\S+)", line)
            if not match:
                break
            name = match.group(1)
            if name not in result["failed"]:
                result["failed"].append(name)

    if not result["failed"]:
        result["failed"] = re.findall(
            r"^\*{3}(?:Failed|Timeout)\*?\s+[\d.]+s\s+(\S+)",
            output,
            flags=re.MULTILINE,
        )
    return result


def _run(
    cmd: list[str], timeout: int, env: dict[str, str] | None = None
) -> tuple[int | None, str]:
    """Run a command in the project root.

    Returns ``(exit_code, output)``; ``exit_code`` is ``None`` when the
    command could not be started or timed out. ``env`` replaces the ambient
    environment when given (used to run the build inside the MSVC developer
    environment).
    """
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError:
        return None, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s: {' '.join(cmd)}"

    output = completed.stdout or ""
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    return completed.returncode, output.strip()


def _vcpkg_root() -> str | None:
    """Locate VCPKG_ROOT.

    Codex launches MCP servers with a filtered environment, so the user's
    VCPKG_ROOT may be absent here even though it is set system-wide. Fall
    back to the persisted Windows environment (HKCU, then HKLM).
    """
    root = os.environ.get("VCPKG_ROOT")
    if root:
        return root

    try:
        import winreg
    except ImportError:
        return None

    locations = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for hive, key_path in locations:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "VCPKG_ROOT")
        except OSError:
            continue
        if value:
            # REG_EXPAND_SZ values may contain %VAR% references.
            return os.path.expandvars(str(value))
    return None


def _merge_env(*environments: dict[str, str]) -> dict[str, str]:
    """Merge environments with later values winning.

    Windows treats environment variable names case-insensitively, but a plain
    ``{**a, **b}`` keeps both ``Path`` and ``PATH``. Those collide in the child
    process's environment block and the wrong one can win (e.g. hiding the
    compiler that VsDevCmd just added), so collapse such keys to a single name.
    """
    key_case: dict[str, str] = {}
    merged: dict[str, str] = {}
    for environment in environments:
        for key, value in environment.items():
            upper = key.upper()
            previous = key_case.get(upper)
            if previous is not None and previous != key:
                merged.pop(previous, None)
            key_case[upper] = key
            merged[key] = value
    return merged


def _program_files_roots() -> list[pathlib.Path]:
    """Return candidate Program Files roots, most-specific first.

    The environment the agent launches us with may omit ``ProgramFiles(x86)``,
    so fall back to the (unfiltered) registry values under
    ``Windows\\CurrentVersion``.
    """
    roots: list[pathlib.Path] = []
    for variable in ("ProgramFiles(x86)", "ProgramW6432", "ProgramFiles"):
        value = os.environ.get(variable)
        if value:
            roots.append(pathlib.Path(value))

    try:
        import winreg
    except ImportError:
        pass
    else:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion",
            ) as key:
                for name in ("ProgramFilesDir (x86)", "ProgramFilesDir"):
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except OSError:
                        continue
                    if value:
                        roots.append(
                            pathlib.Path(os.path.expandvars(str(value)))
                        )
        except OSError:
            pass

    unique: list[pathlib.Path] = []
    seen: set[str] = set()
    for root in roots:
        marker = str(root).lower()
        if marker not in seen:
            seen.add(marker)
            unique.append(root)
    return unique


def _find_vswhere() -> pathlib.Path | None:
    """Locate vswhere.exe without hard-coding a Program Files path."""
    for root in _program_files_roots():
        candidate = root / VSWHERE_SUBPATH
        if candidate.is_file():
            return candidate
    return None


def _find_vs_install() -> pathlib.Path | None:
    """Locate a Visual Studio installation providing the C++ toolset."""
    vswhere = _find_vswhere()
    if vswhere is None:
        return None

    code, output = _run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            VS_CPP_COMPONENT,
            "-property",
            "installationPath",
        ],
        ENV_TIMEOUT,
    )
    if code != 0 or not output.strip():
        return None

    install = output.strip().splitlines()[0].strip()
    return pathlib.Path(install) if install else None


def _msvc_environment() -> tuple[dict[str, str] | None, str]:
    """Reproduce the Visual Studio developer environment.

    cl.exe is never on the system PATH and needs INCLUDE/LIB to compile at
    all, so a plain shell cannot use MSVC. Agents launch us with a filtered
    environment, so activate the developer environment ourselves instead of
    depending on how the agent was started.

    Returns ``(environment, detail)``: on success ``environment`` is the full
    environment produced by ``VsDevCmd.bat`` and ``detail`` is the install
    path; on failure ``environment`` is ``None`` and ``detail`` says why.
    """
    install = _find_vs_install()
    if install is None:
        return None, "no Visual Studio install with the C++ toolset was found"

    dev_cmd = install / "Common7" / "Tools" / "VsDevCmd.bat"
    args = ["-arch=x64", "-host_arch=x64"]
    if not dev_cmd.is_file():
        dev_cmd = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        args = []
    if not dev_cmd.is_file():
        return None, f"neither VsDevCmd.bat nor vcvars64.bat exists under {install}"

    # Run through shell=True: passing this as an argument list makes Python
    # escape the embedded quotes with backslashes, which cmd.exe does not
    # understand, so VsDevCmd.bat would fail with exit code 1.
    command = f'call "{dev_cmd}" {" ".join(args)} >nul && set'
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=ENV_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, f"{dev_cmd.name} timed out after {ENV_TIMEOUT}s"
    if completed.returncode != 0:
        return None, f"{dev_cmd.name} failed (exit code {completed.returncode})"

    output = (completed.stdout or "") + (completed.stderr or "")

    environment: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            environment[key] = value
    if "INCLUDE" not in environment or "LIB" not in environment:
        return None, f"{dev_cmd.name} did not set INCLUDE/LIB"

    return environment, str(install)


def _write_marker(marker: pathlib.Path, state: str, **extra) -> None:
    """Persist the async-run state; failures stay silent (best effort)."""
    data = {"state": state}
    data.update(extra)
    try:
        marker.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _find_running_marker() -> pathlib.Path | None:
    """Return a leftover 'running' marker from a dead server, if any."""
    build_root = REPO_ROOT / "build"
    if not build_root.exists():
        return None
    for marker in sorted(build_root.glob("*-agent*/mcp_ctest.json")):
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("state") == "running":
            return marker
    return None


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ctest together with its child test processes.

    Popen.kill() only terminates ctest itself; on Windows the python
    test processes would survive as orphans, so ask taskkill to take the
    whole process tree down. Both variants are best effort.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()


def _finalize_ctest(handle: dict, code: int, timed_out: bool = False):
    """Close a terminal background run and persist its state.

    Shared by test_status (normal completion) and _start_async_ctest
    (reaping a finished-but-unpolled run before a new start). Returns
    ``(tests, output)`` for payload construction; all I/O is best effort.
    """
    try:
        handle["stream"].close()
    except OSError:
        pass
    try:
        output = handle["log_file"].read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        output = ""
    tests = _parse_ctest(output)
    _write_marker(
        handle["marker"],
        "timeout" if timed_out else "finished",
        exit_code=code,
        run=handle["run"],
        summary=tests["summary"],
    )
    result_line = (
        f"RESULT: FAILED (async ctest, exit code {code}"
        + (", killed after timeout" if timed_out else "")
        + ")"
    )
    try:
        with handle["build_log"].open(
            "a", encoding="utf-8", newline="\n"
        ) as log:
            log.write("\n$ " + " ".join(handle["cmd"]) + "\n")
            log.write(output + "\n" + result_line + "\n")
    except OSError:
        pass
    return tests, output, result_line


def _start_async_ctest(
    test_cmd: list[str],
    build_dir: pathlib.Path,
    build_env: dict[str, str] | None,
    config: str,
    run: str,
    python: bool,
    quiet: bool,
) -> tuple[str, bool]:
    """Start ctest in the background and return immediately."""
    global _ACTIVE_CTEST
    if _ACTIVE_CTEST is not None:
        code = _ACTIVE_CTEST["proc"].poll()
        if code is None:
            return (
                json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "a background ctest is already running; poll "
                            "test_status until it finishes before starting "
                            "another one"
                        ),
                    },
                    indent=2,
                ),
                False,
            )
        # Finished but never polled: reap it (close the stream, persist the
        # marker, append the log) instead of leaking the handle.
        _finalize_ctest(_ACTIVE_CTEST, code)
        _ACTIVE_CTEST = None

    log_file = build_dir / "ctest_async.log"
    marker = build_dir / "mcp_ctest.json"
    try:
        stream = log_file.open("w", encoding="utf-8", errors="replace")
    except OSError as exc:
        return (
            json.dumps(
                {
                    "ok": False,
                    "error": f"cannot open {log_file} for ctest output: {exc}",
                },
                indent=2,
            ),
            False,
        )
    try:
        proc = subprocess.Popen(
            test_cmd,
            cwd=str(REPO_ROOT),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=build_env,
        )
    except OSError as exc:
        stream.close()
        return (
            json.dumps({"ok": False, "error": f"failed to start ctest: {exc}"}),
            False,
        )

    _ACTIVE_CTEST = {
        "proc": proc,
        "stream": stream,
        "cmd": test_cmd,
        "log_file": log_file,
        "marker": marker,
        "build_log": build_dir / "build_and_test.log",
        "config": config,
        "run": run,
        "python": python,
        "quiet": quiet,
        "started_mono": time.monotonic(),
    }
    _write_marker(
        marker,
        "running",
        pid=proc.pid,
        run=run,
        config=config,
        started=time.time(),
        log=log_file.relative_to(REPO_ROOT).as_posix(),
    )
    payload = {
        "ok": True,
        "state": "running",
        "stage": "test-async",
        "config": config,
        "run": run,
        "python": python,
        "build_dir": str(build_dir),
        "async_log": log_file.relative_to(REPO_ROOT).as_posix(),
        "report": (
            "ctest started in the background; poll the test_status tool "
            "until it reports a terminal state"
        ),
    }
    return json.dumps(payload, indent=2), True


def test_status() -> tuple[str, bool]:
    """Report on the background ctest run; see TEST_STATUS_DESCRIPTION."""
    global _ACTIVE_CTEST
    if _ACTIVE_CTEST is None:
        stale = _find_running_marker()
        if stale is not None:
            return (
                json.dumps(
                    {
                        "ok": False,
                        "state": "stale",
                        "marker": str(stale),
                        "error": (
                            "the marker says a ctest is running but this "
                            "server process has no handle to it (the server "
                            "restarted mid-run); rerun the tests"
                        ),
                    },
                    indent=2,
                ),
                False,
            )
        return (
            json.dumps(
                {"ok": False, "error": "no background test run is active"},
                indent=2,
            ),
            False,
        )

    handle = _ACTIVE_CTEST
    elapsed = time.monotonic() - handle["started_mono"]
    code = handle["proc"].poll()
    timed_out = False
    if code is None and elapsed > ASYNC_TEST_TIMEOUT:
        _kill_tree(handle["proc"])
        code = handle["proc"].wait()
        timed_out = True

    if code is None:
        try:
            tail = handle["log_file"].read_text(
                encoding="utf-8", errors="replace"
            )[-2000:]
        except OSError:
            tail = ""
        payload = {
            "ok": True,
            "state": "running",
            "elapsed_s": round(elapsed, 1),
            "run": handle["run"],
            "build_dir": str(handle["log_file"].parent),
            "tail": tail,
        }
        return json.dumps(payload, indent=2), True

    # Terminal state: close the log stream, parse, persist, report.
    _ACTIVE_CTEST = None
    tests, output, result_line = _finalize_ctest(handle, code, timed_out)
    failed = code != 0
    report = (
        result_line
        if failed
        else f"RESULT: SUCCESS (async, {tests['summary']})"
    )
    if not handle["quiet"]:
        report += "\n" + _tail(output)

    payload = {
        "ok": not failed,
        "state": "timeout" if timed_out else "finished",
        "elapsed_s": round(elapsed, 1),
        "config": handle["config"],
        "run": handle["run"],
        "python": handle["python"],
        "build_dir": str(handle["log_file"].parent),
        "tests": tests,
        "report": report,
    }
    if failed:
        payload["first_error"] = _first_error(output)
        payload["tail"] = _tail(output)
    payload["log_path"] = handle["build_log"].relative_to(REPO_ROOT).as_posix()
    return json.dumps(payload, indent=2), not failed


def build_and_test(
    config: str = "Release",
    target: str = "all",
    run: str = "unit",
    python: bool = False,
    quiet: bool = True,
    wait: bool | None = None,
) -> tuple[str, bool]:
    """Configure, build and optionally run the project.

    Returns ``(json_text, ok)``.
    """
    if config not in ("Debug", "Release"):
        return (
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"invalid config {config!r}: expected Debug or Release"
                    ),
                },
                indent=2,
            ),
            False,
        )
    if target not in ("all", "fvm_tests", "pyfvm"):
        return (
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"invalid target {target!r}: expected all, fvm_tests "
                        "or pyfvm"
                    ),
                },
                indent=2,
            ),
            False,
        )
    if run == "tests":  # deprecated alias from the pre-layering protocol
        run = "all"
    if run not in RUN_MODES:
        return (
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"invalid run {run!r}: expected none, unit, "
                        "application or all"
                    ),
                },
                indent=2,
            ),
            False,
        )
    if run == "application" and not python:
        return (
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "invalid combination: run=application requires "
                        "python=true (the application tests are registered "
                        "in the FVM_BUILD_PYTHON block)"
                    ),
                },
                indent=2,
            ),
            False,
        )
    if target == "pyfvm" and not python:
        return (
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "invalid combination: target=pyfvm requires "
                        "python=true"
                    ),
                },
                indent=2,
            ),
            False,
        )
    if wait is not None and not isinstance(wait, bool):
        return (
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"invalid wait {wait!r}: expected true, false or "
                        "omitted"
                    ),
                },
                indent=2,
            ),
            False,
        )

    build_dir = REPO_ROOT / "build" / f"{config}-agent{'-py' if python else ''}"
    sections = [f"project: {REPO_ROOT}", f"build dir: {build_dir}"]
    transcript: list[str] = []
    failed = False
    stage = "configure"
    test_summary: str | None = None
    tests: dict | None = None
    log_file = build_dir / "build_and_test.log"
    log_path: str | None = None

    def write_log() -> None:
        """Persist the complete stage transcript; failures stay silent."""
        nonlocal log_path
        if not log_file.parent.exists():
            return
        try:
            with log_file.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(transcript) + "\n")
            log_path = log_file.relative_to(REPO_ROOT).as_posix()
        except OSError:
            log_path = None

    # --- Build environment ---
    # Agents launch us with a filtered environment, so activate MSVC here
    # instead of depending on the shell that started the agent.
    build_env: dict[str, str] | None = None
    if os.name == "nt":
        msvc_env, detail = _msvc_environment()
        if msvc_env is None:
            sections.append(
                "WARNING: could not activate the MSVC developer environment "
                f"({detail}); falling back to the ambient environment"
            )
        else:
            build_env = _merge_env(os.environ, msvc_env)
            sections.append(f"MSVC developer environment activated: {detail}")

    # --- Configure ---
    configure_cmd = [
        "cmake",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={config}",
    ]
    if os.name == "nt":
        # vcpkg derives the default triplet from the host architecture, which
        # is absent from the filtered environment agents launch us with; it
        # then warns and silently continues without itself, so doctest/eigen3
        # are never found. Pin the triplet instead of relying on detection.
        configure_cmd.append("-DVCPKG_TARGET_TRIPLET=x64-windows")
    if build_env is not None:
        # Pin the compiler too, so the project is built with the same toolset
        # as the vcpkg dependencies rather than whatever is first on PATH.
        configure_cmd.append("-DCMAKE_CXX_COMPILER=cl")
    vcpkg_root = _vcpkg_root()
    if vcpkg_root:
        if not os.environ.get("VCPKG_ROOT"):
            sections.append(
                f"VCPKG_ROOT is not in the process environment; resolved "
                f"{vcpkg_root} from the persisted Windows environment."
            )
        configure_cmd.append(
            "-DCMAKE_TOOLCHAIN_FILE="
            + str(
                pathlib.Path(vcpkg_root)
                / "scripts"
                / "buildsystems"
                / "vcpkg.cmake"
            )
        )
    if python:
        configure_cmd.append("-DFVM_BUILD_PYTHON=ON")
    else:
        sections.append(
            "WARNING: VCPKG_ROOT is not set; configuring without the vcpkg "
            "toolchain file (eigen3/doctest will not be found)."
        )
    sections.append("\n$ " + " ".join(configure_cmd))
    transcript.append("$ " + " ".join(configure_cmd))
    code, output = _run(configure_cmd, CONFIGURE_TIMEOUT, build_env)
    sections.append(_tail(output))
    transcript.append(output)
    if code != 0:
        sections.append(f"\nCONFIGURE FAILED (exit code {code})")

    write_log()

    if code != 0:
        sections.append("\nRESULT: FAILED")
        transcript.append("\nRESULT: FAILED (configure)")
        write_log()
        report = "\n".join(sections) if not quiet else (
            f"CONFIGURE FAILED (exit code {code})\nRESULT: FAILED"
        )
        payload: dict = {
            "ok": False,
            "config": config,
            "target": target,
            "run": run,
            "python": python,
            "build_dir": str(build_dir),
            "stage": stage,
            "first_error": _first_error(output),
            "tail": _tail(output),
            "report": report,
        }
        if log_path is not None:
            payload["log_path"] = log_path
        return json.dumps(payload, indent=2), False

    stage = "build"
    write_log()
    # --- Build ---
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
    if target != "all":
        build_cmd += ["--target", target]
    sections.append("\n$ " + " ".join(build_cmd))
    transcript.append("\n$ " + " ".join(build_cmd))
    code, output = _run(build_cmd, BUILD_TIMEOUT, build_env)
    sections.append(_tail(output))
    transcript.append(output)
    write_log()
    if code != 0:
        sections.append(f"\nBUILD FAILED (exit code {code})")
        sections.append("\nRESULT: FAILED")
        transcript.append(f"\nRESULT: FAILED (build, exit code {code})")
        write_log()
        report = "\n".join(sections) if not quiet else (
            f"BUILD FAILED (exit code {code})\nRESULT: FAILED"
        )
        payload = {
            "ok": False,
            "config": config,
            "target": target,
            "run": run,
            "python": python,
            "build_dir": str(build_dir),
            "stage": stage,
            "first_error": _first_error(output),
            "tail": _tail(output),
            "report": report,
        }
        if log_path is not None:
            payload["log_path"] = log_path
        return json.dumps(payload, indent=2), False
    sections.append(f"\nBUILD SUCCEEDED ({config}, target: {target})")

    stage = "test" if run != "none" and target != "pyfvm" else "done"
    # --- Run ---
    if run != "none":
        if target == "pyfvm":
            sections.append(
                f"\nSKIPPED tests: fvm_tests was not built (target={target})"
            )
        else:
            test_cmd = [
                "ctest",
                "--test-dir",
                str(build_dir),
                "--output-on-failure",
                "-C",
                config,
            ]
            if run == "unit":
                test_cmd += ["-LE", "application"]
            elif run == "application":
                test_cmd += ["-L", "application"]
            sections.append("\n$ " + " ".join(test_cmd))
            transcript.append("\n$ " + " ".join(test_cmd))

            sync = wait if wait is not None else run in ("none", "unit")
            if not sync:
                sections.append(
                    "\nctest runs in the background (wait=false); poll the "
                    "test_status tool for the result"
                )
                transcript.append(
                    "\n$ ctest launched in the background; see "
                    "ctest_async.log / test_status"
                )
                write_log()
                return _start_async_ctest(
                    test_cmd, build_dir, build_env, config, run, python, quiet
                )

            code, output = _run(test_cmd, RUN_TIMEOUT, build_env)
            sections.append(_tail(output))
            transcript.append(output)
            write_log()
            tests = _parse_ctest(output)
            if code != 0:
                failed = True
                sections.append(f"\nTESTS FAILED (exit code {code})")
            else:
                sections.append("\nALL TESTS PASSED")
                test_summary = tests["summary"] or "all tests passed"

    if not failed and quiet:
        summary = [
            f"BUILD OK: {config}, target={target}, "
            f"python={python} ({build_dir.name})",
            f"TESTS: {test_summary or 'not run'}",
            "RESULT: SUCCESS (quiet=true; full transcript with quiet=false)",
        ]
        payload = {
            "ok": True,
            "config": config,
            "target": target,
            "run": run,
            "python": python,
            "build_dir": str(build_dir),
            "stage": "done",
            "report": "\n".join(summary),
        }
        if tests is not None:
            payload["tests"] = tests
        if log_path is not None:
            payload["log_path"] = log_path
        return json.dumps(payload, indent=2), True

    result_line = "RESULT: FAILED" if failed else "RESULT: SUCCESS"
    sections.append("\n" + result_line)
    transcript.append("\n" + result_line)
    write_log()
    payload = {
        "ok": not failed,
        "config": config,
        "target": target,
        "run": run,
        "python": python,
        "build_dir": str(build_dir),
        "stage": stage,
        "report": "\n".join(sections),
    }
    if tests is not None:
        payload["tests"] = tests
    if failed:
        payload["first_error"] = _first_error(output)
        payload["tail"] = _tail(output)
    if log_path is not None:
        payload["log_path"] = log_path
    return json.dumps(payload, indent=2), not failed


# ---------------------------------------------------------------------------
# Minimal MCP plumbing (newline-delimited JSON-RPC 2.0 over stdio)
# ---------------------------------------------------------------------------


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
        return  # notification (e.g. notifications/initialized): no answer

    if method == "initialize":
        _reply(
            request_id,
            {
                "protocolVersion": params.get("protocolVersion")
                or DEFAULT_PROTOCOL_VERSION,
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
                    {
                        "name": "build_and_test",
                        "description": TOOL_DESCRIPTION,
                        "inputSchema": TOOL_SCHEMA,
                    },
                    {
                        "name": "test_status",
                        "description": TEST_STATUS_DESCRIPTION,
                        "inputSchema": TEST_STATUS_SCHEMA,
                    },
                ]
            },
        )
    elif method == "tools/call":
        name = params.get("name")
        if name not in ("build_and_test", "test_status"):
            _reply(
                request_id,
                error={"code": -32602, "message": f"unknown tool: {name}"},
            )
            return
        arguments = params.get("arguments") or {}
        try:
            if name == "test_status":
                text, ok = test_status()
            else:
                text, ok = build_and_test(
                    config=arguments.get("config") or "Release",
                    target=arguments.get("target") or "all",
                    run=arguments.get("run") or "unit",
                    python=bool(arguments.get("python")),
                    quiet=bool(arguments.get("quiet", True)),
                    wait=arguments.get("wait"),
                )
        except Exception as exc:  # never take the server down
            text, ok = f"{name} crashed: {exc!r}", False
        _reply(
            request_id,
            {"content": [{"type": "text", "text": text}], "isError": not ok},
        )
    else:
        _reply(
            request_id,
            error={"code": -32601, "message": f"method not found: {method}"},
        )


def main() -> int:
    # Keep the framing identical on every platform: JSON-RPC messages are
    # separated by a single "\n" (Windows text mode would emit "\r\n").
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
