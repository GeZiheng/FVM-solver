#!/usr/bin/env python3
"""Stdio MCP server exposing this project's build-and-test workflow.

Speaks just enough of the Model Context Protocol (newline-delimited
JSON-RPC 2.0 over stdin/stdout) to expose one tool, ``build_and_test``.
It uses only the Python standard library, so it runs without installing
an MCP SDK or reaching the network.

The tool mirrors the opencode tool it replaces
(``.opencode/tools/build-and-test.ts``): configure with CMake + Ninja into
``build/<config>-opencode``, build ``fvm_solver`` and/or ``fvm_tests``, then
optionally run ctest and/or the ``fvm_solver`` demo.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

MAX_OUTPUT_CHARS = 8000
SERVER_NAME = "build-and-test"
SERVER_VERSION = "1.0.0"
# Used only when the client does not request a version; otherwise the
# requested version is echoed back so the client keeps the negotiation.
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

CONFIGURE_TIMEOUT = 1800
BUILD_TIMEOUT = 1800
RUN_TIMEOUT = 1800

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

TOOL_DESCRIPTION = (
    "Configure (cmake -G Ninja) and build this FVM-solver project in Debug or "
    "Release mode (default Release), building fvm_solver and/or fvm_tests "
    "(default both), then optionally run ctest and/or the fvm_solver demo. Use "
    "this instead of invoking cmake/ctest/ninja directly for project builds. "
    "When the user directly requests a build or test, confirm the options "
    "(config, target, run) with the user first unless they already stated them. "
    "When building as part of a code modification workflow, proceed with the "
    "defaults (config=Release, target=all, run=tests)."
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
            "enum": ["all", "fvm_solver", "fvm_tests"],
            "default": "all",
            "description": (
                "Which target(s) to build: all (default), fvm_solver, or fvm_tests."
            ),
        },
        "run": {
            "type": "string",
            "enum": ["none", "tests", "solver", "both"],
            "default": "tests",
            "description": (
                "What to run after a successful build: none, tests (ctest, "
                "requires fvm_tests), solver (run the fvm_solver demo, requires "
                "fvm_solver), or both (default tests)."
            ),
        },
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


def _run(cmd: list[str], timeout: int) -> tuple[int | None, str]:
    """Run a command in the project root.

    Returns ``(exit_code, output)``; ``exit_code`` is ``None`` when the
    command could not be started or timed out.
    """
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
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


def build_and_test(
    config: str = "Release", target: str = "all", run: str = "tests"
) -> tuple[str, bool]:
    """Configure, build and optionally run the project.

    Returns ``(report, ok)``.
    """
    if config not in ("Debug", "Release"):
        return f"invalid config {config!r}: expected Debug or Release", False
    if target not in ("all", "fvm_solver", "fvm_tests"):
        return (
            f"invalid target {target!r}: expected all, fvm_solver or fvm_tests",
            False,
        )
    if run not in ("none", "tests", "solver", "both"):
        return f"invalid run {run!r}: expected none, tests, solver or both", False

    build_dir = REPO_ROOT / "build" / f"{config}-opencode"
    sections = [f"project: {REPO_ROOT}", f"build dir: {build_dir}"]
    failed = False

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
    else:
        sections.append(
            "WARNING: VCPKG_ROOT is not set; configuring without the vcpkg "
            "toolchain file (eigen3/doctest will not be found)."
        )
    sections.append("\n$ " + " ".join(configure_cmd))
    code, output = _run(configure_cmd, CONFIGURE_TIMEOUT)
    sections.append(_tail(output))
    if code != 0:
        sections.append(f"\nCONFIGURE FAILED (exit code {code})")
        return "\n".join(sections), False

    # --- Build ---
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
    if target != "all":
        build_cmd += ["--target", target]
    sections.append("\n$ " + " ".join(build_cmd))
    code, output = _run(build_cmd, BUILD_TIMEOUT)
    sections.append(_tail(output))
    if code != 0:
        sections.append(f"\nBUILD FAILED (exit code {code})")
        return "\n".join(sections), False
    sections.append(f"\nBUILD SUCCEEDED ({config}, target: {target})")

    # --- Run ---
    if run in ("tests", "both"):
        if target == "fvm_solver":
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
            sections.append("\n$ " + " ".join(test_cmd))
            code, output = _run(test_cmd, RUN_TIMEOUT)
            sections.append(_tail(output))
            if code != 0:
                failed = True
                sections.append(f"\nTESTS FAILED (exit code {code})")
            else:
                sections.append("\nALL TESTS PASSED")

    if run in ("solver", "both"):
        if target == "fvm_tests":
            sections.append(
                f"\nSKIPPED solver: fvm_solver was not built (target={target})"
            )
        else:
            suffix = ".exe" if os.name == "nt" else ""
            solver_exe = build_dir / f"fvm_solver{suffix}"
            sections.append(f"\n$ {solver_exe} (cwd: {build_dir})")
            try:
                completed = subprocess.run(
                    [str(solver_exe)],
                    cwd=str(build_dir),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=RUN_TIMEOUT,
                )
                output = completed.stdout or ""
                if completed.stderr:
                    output += ("\n" if output else "") + completed.stderr
                sections.append(_tail(output.strip()))
                if completed.returncode != 0:
                    failed = True
                    sections.append(
                        f"\nSOLVER FAILED (exit code {completed.returncode})"
                    )
                else:
                    sections.append("\nSOLVER RUN SUCCEEDED")
            except FileNotFoundError:
                failed = True
                sections.append(f"\nSOLVER FAILED: {solver_exe} not found")
            except subprocess.TimeoutExpired:
                failed = True
                sections.append(f"\nSOLVER FAILED: timed out after {RUN_TIMEOUT}s")

    sections.append("\n" + ("RESULT: FAILED" if failed else "RESULT: SUCCESS"))
    return "\n".join(sections), not failed


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
                    }
                ]
            },
        )
    elif method == "tools/call":
        name = params.get("name")
        if name != "build_and_test":
            _reply(
                request_id,
                error={"code": -32602, "message": f"unknown tool: {name}"},
            )
            return
        arguments = params.get("arguments") or {}
        try:
            text, ok = build_and_test(
                config=arguments.get("config") or "Release",
                target=arguments.get("target") or "all",
                run=arguments.get("run") or "tests",
            )
        except Exception as exc:  # never take the server down
            text, ok = f"build_and_test crashed: {exc!r}", False
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
