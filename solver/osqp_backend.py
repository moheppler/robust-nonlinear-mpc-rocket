"""Single source of truth for choosing the OSQP backend.

The compiled codegen extension (``osqp_generated``) is the fast, default path used for the
paper's timing results. When it is not built/installed (no C compiler, or the user opted out),
the pure-Python ``osqp`` wheel is a backup - same QP, slower - but it is used only if the user
actively opts in (``--stock-osqp`` / ``set_force_stock(True)``). Without that opt-in,
``select_backend`` refuses with a clear error rather than silently downgrading.
"""
import glob
import sys
from pathlib import Path

CODEGEN_DIR = Path(__file__).resolve().parents[1] / "build" / "osqp_fast"

_force_stock = False
_available_cache = None  # None = not yet probed
_announced = False


class FastSolverRequiredError(RuntimeError):
    """Raised when the compiled solver is unavailable and stock OSQP was not explicitly requested."""


def set_force_stock(flag: bool) -> None:
    global _force_stock, _available_cache
    _force_stock = bool(flag)
    _available_cache = None  # re-probe next time


def _extension_importable() -> bool:
    """True iff the compiled ``osqp_generated`` extension can actually be imported.

    Ignores ``_force_stock`` - this is a raw availability probe. Side effect: ensures
    CODEGEN_DIR is on sys.path so a later ``__import__("osqp_generated")`` (in qp_jit.py's
    codegen backend) resolves even before the package is pip-installed.
    """
    global _available_cache
    if _available_cache is not None:
        return _available_cache

    if CODEGEN_DIR.is_dir() and str(CODEGEN_DIR) not in sys.path:
        sys.path.insert(0, str(CODEGEN_DIR))
    try:
        import osqp_generated  # noqa: F401
        _available_cache = True
    except Exception:
        # fall back to a raw .so/.pyd probe in the build dir
        matches = glob.glob(str(CODEGEN_DIR / "osqp_generated*.so")) + \
            glob.glob(str(CODEGEN_DIR / "osqp_generated*.pyd"))
        _available_cache = bool(matches)
    return _available_cache


def fast_solver_available() -> bool:
    """True iff the compiled ``osqp_generated`` extension can be imported and stock is not forced."""
    if _force_stock:
        return False
    return _extension_importable()


def select_backend() -> str:
    """Pick the OSQP backend, requiring active consent for the slower pure-Python path.

    Returns ``"osqp_codegen"`` when the compiled extension is available and stock wasn't
    forced. Returns ``"osqp"`` only if the user explicitly opted in (``--stock-osqp`` /
    ``set_force_stock(True)``). Otherwise raises ``FastSolverRequiredError`` - a fresh
    checkout without the compiled solver must not silently run the slow path.
    """
    if fast_solver_available():
        return "osqp_codegen"
    if _force_stock:
        return "osqp"
    raise FastSolverRequiredError(
        "Compiled fast solver not found in build/osqp_fast, and the pure-Python fallback was "
        "not requested.\n"
        "  Either build the fast solver:   rnmpc build-fast <system>\n"
        "  Or explicitly opt into the slower pure-Python backend:   rnmpc <system> --run --stock-osqp"
    )


def announce_backend_once(backend: str) -> None:
    global _announced
    if backend == "osqp" and not _announced:
        _announced = True
        print(
            "\n[rnmpc] Compiled fast solver not found - running the pure-Python OSQP backend "
            "(significantly slower).\n"
            "        The paper's timing results require the compiled solver. To build it, "
            "install a C compiler + CMake\n"
            "        (see README - Prerequisites) and run:  rnmpc build-fast <system>\n"
        )
