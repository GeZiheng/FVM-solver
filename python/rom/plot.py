"""Optional matplotlib diagnostics for POD results.

matplotlib is imported inside each plotting function, never at module
import time: the rest of the ``rom`` package, the test suite and the CI
path must work without matplotlib installed.  When it is missing, the
plotting functions raise ``RuntimeError`` with an installation hint.

Both functions accept any POD-like result object exposing
``singular_values`` / ``energy`` / ``cumulative_energy`` /
``reconstruction_error(r)`` (a :class:`rom.pod.PodResult` qualifies)
and draw onto the given matplotlib Axes, creating one when omitted.
Each returns the Axes it drew on.
"""

from __future__ import annotations


def _import_matplotlib():
    """Import matplotlib.pyplot lazily or fail with an install hint."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for rom.plot but is not installed; "
            "install it with: pip install matplotlib") from exc
    return plt


def plot_energy_spectrum(result, ax=None):
    """Plot the POD energy spectrum.

    Draws the per-mode normalized energy ``e_k`` on a log-scaled y
    axis plus the cumulative energy (0-1) on a secondary right axis.

    Args:
        result: POD-like result with ``energy`` and ``cumulative_energy``.
        ax: optional matplotlib Axes to draw on.

    Returns:
        The matplotlib Axes that was drawn on.
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    modes = list(range(1, result.singular_values.size + 1))
    ax.plot(modes, result.energy, "o-", label="energy $e_k$")
    ax.set_yscale("log")
    ax.set_xlabel("POD mode index")
    ax.set_ylabel("normalized energy $e_k$")
    twin = ax.twinx()
    twin.plot(modes, result.cumulative_energy, "s--", color="tab:orange",
              label="cumulative")
    twin.set_ylim(0.0, 1.05)
    twin.set_ylabel("cumulative energy")
    ax.set_title("POD energy spectrum")
    return ax


def plot_reconstruction_error(result, ax=None):
    """Plot the POD rank-r reconstruction error ``err(r)``.

    Evaluates ``reconstruction_error(r)`` for r = 0..r_full and draws
    it on a log-scaled y axis.

    Args:
        result: POD-like result with ``singular_values`` and
            ``reconstruction_error(r)``.
        ax: optional matplotlib Axes to draw on.

    Returns:
        The matplotlib Axes that was drawn on.
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    ranks = list(range(0, result.singular_values.size + 1))
    errors = [result.reconstruction_error(r) for r in ranks]
    ax.semilogy(ranks, errors, "o-")
    ax.set_xlabel("number of POD modes r")
    ax.set_ylabel("reconstruction error err(r)")
    ax.set_title("POD reconstruction error")
    return ax
