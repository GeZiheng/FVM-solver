#!/usr/bin/env bash
#
# Create an independent OpenFOAM case from a template, run it, and summarise
# the log.  The configuration is baked into the directory name so that results
# from different settings can never be mixed up (a mistake that has produced
# wrong conclusions in this project before).
#
# Usage:
#   make_of_case.sh --template <case> --root <dir> --name <label> --nc <N>
#                   [--dt <s>] [--end <t>] [--dx <m>] [--solver <name>]
#                   [--setup-only]
#
# --setup-only: set up the case (copy + patch) and exit before running.
# dt/dx/nc/solver are recorded in mcp_run so a launcher can rebuild the
# summary parameters; the caller is responsible for running foamRun and
# writing its exit code to .exitcode.  (Background processes spawned from a
# `wsl.exe -e` invocation do not survive it, so the caller must keep a
# wsl.exe process alive for the run - this is what the MCP server does.)
#
# Requires an OpenFOAM environment (source etc/bashrc) - pass FOAM_BASHRC to
# override the default ~/OpenFOAM/OpenFOAM-14/etc/bashrc.
#
set -euo pipefail

template=""
root="$HOME/OpenFOAM/$USER-14/run"
label=""
nc=""
dt=""
end=""
dx=""
solver="incompressibleFluid"
setup_only=""
foam_bashrc="${FOAM_BASHRC:-$HOME/OpenFOAM/OpenFOAM-14/etc/bashrc}"

usage() { sed -n '3,14p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --template) template="$2"; shift 2 ;;
        --root)     root="$2"; shift 2 ;;
        --name)     label="$2"; shift 2 ;;
        --nc)       nc="$2"; shift 2 ;;
        --dt)       dt="$2"; shift 2 ;;
        --end)      end="$2"; shift 2 ;;
        --dx)       dx="$2"; shift 2 ;;
        --solver)   solver="$2"; shift 2 ;;
        --setup-only) setup_only="yes"; shift ;;
        -h|--help)  usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[ -n "$template" ] || { echo "--template is required" >&2; usage 1; }
[ -d "$template" ] || { echo "no such template case: $template" >&2; exit 1; }
[ -n "$label" ]    || { echo "--name is required" >&2; usage 1; }
[ -n "$nc" ]       || { echo "--nc is required" >&2; usage 1; }

stamp=$(date +%m%d-%H%M%S)
case_name="${label}_nc${nc}"
[ -n "$dt" ]  && case_name="${case_name}_dt${dt}"
[ -n "$end" ] && case_name="${case_name}_t${end}"
case_dir="$root/${case_name}_${stamp}"

mkdir -p "$case_dir"
cp -r "$template/0"        "$case_dir/0"
cp -r "$template/constant" "$case_dir/constant"
cp -r "$template/system"   "$case_dir/system"

# Patch dictionaries in place.  The patterns tolerate leading whitespace and
# any current value, including the trailing ';'.
patch_value() {
    local file="$1" key="$2" value="$3"
    sed -i -E "s|^([[:space:]]*${key}[[:space:]]+)[^;]*;|\1${value};|" "$file"
    grep -qE "^[[:space:]]*${key}[[:space:]]+${value};" "$file" \
        || { echo "failed to set ${key} in ${file}" >&2; exit 1; }
}

if [ -n "$end" ]; then
    patch_value "$case_dir/system/controlDict" endTime "$end"
    patch_value "$case_dir/system/controlDict" writeInterval "$end"
fi
[ -n "$dt" ] && patch_value "$case_dir/system/controlDict" deltaT "$dt"
patch_value "$case_dir/system/fvSolution" nCorrectors "$nc"

echo "case: $case_dir"
echo "  nCorrectors=$nc  deltaT=${dt:-<template>}  endTime=${end:-<template>}  solver=$solver"
echo "  (config is part of the directory name on purpose - never mix outputs from different settings)"

if [ -n "$setup_only" ]; then
    {
        [ -n "$dt" ] && echo "dt=$dt"
        [ -n "$dx" ] && echo "dx=$dx"
        echo "nc=$nc"
        echo "solver=$solver"
    } > "$case_dir/mcp_run"
    echo "setup-only: case ready (run foamRun separately)"
    exit 0
fi

set +u
# shellcheck disable=SC1090
source "$foam_bashrc" >/dev/null
set -u

log="$case_dir/log.run"
( cd "$case_dir" && foamRun -solver "$solver" > "$log" 2>&1 )
status=$?

if grep -q "FOAM FATAL" "$log"; then
    echo "OpenFOAM reported a fatal error; see $log" >&2
    exit 1
fi

summary_dir="$(cd "$(dirname "$0")" && pwd)"
if command -v python3 >/dev/null && [ -f "$summary_dir/of_log_summary.py" ]; then
    if [ -n "$dt" ] && [ -n "$dx" ]; then
        python3 "$summary_dir/of_log_summary.py" "$log" --dt "$dt" --dx "$dx"
    else
        python3 "$summary_dir/of_log_summary.py" "$log"
    fi
else
    echo "--- last lines of $log ---"
    tail -20 "$log"
fi

exit "$status"
