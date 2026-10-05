#!/usr/bin/env python3
"""Summarise an OpenFOAM solver log into a per-step table.

Why this exists: the log mixes per-step diagnostics with solver lines, and the
line ordering is easy to misread.  In particular OpenFOAM prints the Courant
number *before* the ``Time = ...`` header of the step it refers to (it is
computed from the field entering that step), while the linear-solver lines come
*after* the header.  Hand-rolled ``grep | awk`` sampling has produced step
indices larger than the number of steps, so this script builds an explicit
per-step record instead.

Usage:
    of_log_summary.py LOG [--dt S] [--dx M] [--rows N]

``--dt`` and ``--dx`` convert ``Co_max`` into ``max|U| = Co_max * dx / dt``
(uniform Cartesian mesh with square cells of size ``dx``), which is a free
per-step proxy for the field magnitude.

The ``res_*`` columns report the *last* occurrence within the step, i.e. for a
PISO run with ``nCorrectors = 2`` that is the second pressure solve - which is
exactly the number you want when comparing corrector behaviour.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field

TIME_RE = re.compile(r"^Time = (\S+)s\s*$")
CO_RE = re.compile(r"^Courant Number mean: (\S+) max: (\S+)\s*$")
RES_RE = re.compile(r"Solving for (\w+), Initial residual = (\S+), Final residual = \S+, No Iterations (\d+)")
CONT_RE = re.compile(r"time step continuity errors : sum local = (\S+), global = \S+")


@dataclass
class Step:
    time: str
    co_max: float | None = None
    residuals: dict[str, float] = field(default_factory=dict)
    iterations: dict[str, int] = field(default_factory=dict)
    continuity: float | None = None


def parse(path: str) -> list[Step]:
    steps: list[Step] = []
    pending_co: float | None = None
    current: Step | None = None

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")

            match = CO_RE.match(line)
            if match:
                pending_co = float(match.group(2))
                continue

            match = TIME_RE.match(line)
            if match:
                current = Step(time=match.group(1), co_max=pending_co)
                pending_co = None
                steps.append(current)
                continue

            if current is None:
                continue

            match = RES_RE.search(line)
            if match:
                current.residuals[match.group(1)] = float(match.group(2))
                current.iterations[match.group(1)] = int(match.group(3))
                continue

            match = CONT_RE.search(line)
            if match:
                current.continuity = float(match.group(1))

    return steps


def fmt(value: float | None, width: int = 11, prec: int = 4) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    if value == 0.0:
        return f"{'0':>{width}}"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:>{width}.3e}"
    return f"{value:>{width}.{prec}g}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--dt", type=float, default=None, help="time step (for Co -> max|U|)")
    parser.add_argument("--dx", type=float, default=None, help="cell size (for Co -> max|U|)")
    parser.add_argument("--rows", type=int, default=12, help="max data rows to print")
    args = parser.parse_args(argv)

    steps = parse(args.log)
    if not steps:
        print(f"{args.log}: no 'Time = ' records found", file=sys.stderr)
        return 1

    field_names = sorted({name for step in steps for name in step.residuals})
    convert = args.dt is not None and args.dx is not None

    header = f"{'t':>8} {'Co_max':>11}"
    if convert:
        header += f" {'max|U|':>11}"
    for name in field_names:
        header += f" {('res_' + name):>11}"
    header += f" {'cont(local)':>11}"
    print(header)
    print("-" * len(header))

    stride = max(1, len(steps) // max(1, args.rows))
    for index, step in enumerate(steps):
        if index % stride and index != len(steps) - 1:
            continue
        row = f"{step.time:>8} {fmt(step.co_max)}"
        if convert:
            row += f" {fmt(None if step.co_max is None else step.co_max * args.dx / args.dt)}"
        for name in field_names:
            row += f" {fmt(step.residuals.get(name))}"
        row += f" {fmt(step.continuity)}"
        print(row)

    print()
    print(f"steps: {len(steps)}   t in [{steps[0].time}, {steps[-1].time}]")
    if convert:
        values = [(step.time, step.co_max * args.dx / args.dt) for step in steps if step.co_max]
        if len(values) >= 2:
            (t0, v0), (t1, v1) = values[0], values[-1]
            if v0 > 0 and v1 > v0:
                per_step = math.exp(math.log(v1 / v0) / (len(values) - 1))
                print(f"max|U|: {v0:.4g} (t={t0}) -> {v1:.4g} (t={t1});  "
                      f"mean per-step growth = {per_step:.3f}")
            if v1 > 1e3 * v0:
                print("=> field magnitude has grown by >1e3 over the run: the case is diverging.")
            elif values[-1][1] < 0.5 * values[len(values) // 2][1]:
                print("=> field magnitude is decreasing.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
