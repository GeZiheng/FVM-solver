#!/usr/bin/env python3
"""Stdio MCP server exposing this project's build-and-test workflow.

Speaks just enough of the Model Context Protocol (newline-delimited
JSON-RPC 2.0 over stdin/stdout) to expose one tool, ``build_and_test``.
It uses only the Python standard library, so it runs without installing
an MCP SDK or reaching the network.

Shared by Codex (registered in ``.codex/config.toml``) and opencode
(reached through the ``mcp`` block in ``opencode.json``), so both agents
drive the same implementation. The tool configures with CMake + Ninja into
``build/<config>-agent``, builds ``fvm_solver`` and/or ``fvm_tests``, then
optionally runs ctest and/or the ``fvm_solver`` demo.
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
ENV_TIMEOUT = 300

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Used to reproduce the Visual Studio developer environment; see
# _msvc_environment(). Deliberately hard-coded rather than read from
# os.environ: the variable that would point at it (ProgramFiles(x86)) is
# filtered out of the environment the agent launches us with.
VSWHERE = pathlib.Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
)
VS_CPP_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

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


def _find_vs_install() -> pathlib.Path | None:
    """Locate a Visual Studio installation providing the C++ toolset."""
    if not VSWHERE.is_file():
        return None

    code, output = _run(
        [
            str(VSWHERE),
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

    build_dir = REPO_ROOT / "build" / f"{config}-agent"
    sections = [f"project: {REPO_ROOT}", f"build dir: {build_dir}"]
    failed = False

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
            build_env = {**os.environ, **msvc_env}
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
    else:
        sections.append(
            "WARNING: VCPKG_ROOT is not set; configuring without the vcpkg "
            "toolchain file (eigen3/doctest will not be found)."
        )
    sections.append("\n$ " + " ".join(configure_cmd))
    code, output = _run(configure_cmd, CONFIGURE_TIMEOUT, build_env)
    sections.append(_tail(output))
    if code != 0:
        sections.append(f"\nCONFIGURE FAILED (exit code {code})")
        return "\n".join(sections), False

    # --- Build ---
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
    if target != "all":
        build_cmd += ["--target", target]
    sections.append("\n$ " + " ".join(build_cmd))
    code, output = _run(build_cmd, BUILD_TIMEOUT, build_env)
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
            code, output = _run(test_cmd, RUN_TIMEOUT, build_env)
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
                    env=build_env,
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
