"""Codegen + compile + install the fast OSQP solver for a given system.

Replaces the three former generate_osqp_*.py scripts. The generated CMake/setup files come
from osqp's codegen templates; we patch them here (pin pybind11, relax the Windows generator)
because build/ is wiped and regenerated on every codegen.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from dyn.LTV import LTV
from dyn.pendulum import Pendulum
from dyn.quadrotor import Quadrotor
from dyn.rocket import Rocket
from dyn.rocket_direct import RocketDirect
from solver.osqp_backend import CODEGEN_DIR
from solver.qp_jit import QP

PYBIND11_TAG = "v2.13.6"

SYSTEM_MODELS = {
    "pendulum": Pendulum,
    "quadrotor": Quadrotor,
    "rocket": Rocket,
    "rocket-compare": Rocket,
    "rocket-direct": RocketDirect,
}


def _make_model(system: str):
    m = SYSTEM_MODELS[system]()
    # Pendulum's experiment sets these explicitly; mirror them so the codegen sparsity
    # matches what the experiment builds. Other systems carry their own defaults.
    if system == "pendulum":
        m.E = 0.003 * np.eye(m.nx)
        m.dt = 0.05
        m.replace_constraints(
            10 * np.ones(m.nx), -10 * np.ones(m.nx),
            5 * np.ones(m.nu), -5 * np.ones(m.nu),
            10 * np.ones(m.nx), -10 * np.ones(m.nx),
        )
    return m


def codegen(system: str, N: int | None = None, smoke_test: bool = False,
            model: "RocketDirect | Rocket | Quadrotor | Pendulum | None" = None) -> Path:
    if system not in SYSTEM_MODELS:
        raise ValueError(f"unknown system {system!r}; choose from {sorted(SYSTEM_MODELS)}")
    # Must match each expe/main_<system>.py's generate(): smoke_test always uses N=3,
    # otherwise the default is N=15. A solver built for the wrong N fails loudly at
    # solve time (qp_jit.py's CSC-pattern check) rather than silently, but it's cheaper
    # to just keep this in sync.
    N = 3 if smoke_test else (15 if N is None else int(N))
    m = model if model is not None else _make_model(system)
    Q = np.eye(m.nx)
    R = np.eye(m.nu)
    Qf = 10 * np.eye(m.nx)
    m_ltv = LTV(m, N)
    qp = QP(N, Q, R, m_ltv, Qf, backend="osqp")  # stock backend only to export the code
    qp.export_osqp_solver(str(CODEGEN_DIR), extension_name="osqp_generated")
    patch_generated(CODEGEN_DIR)
    return CODEGEN_DIR


def patch_generated(build_dir: Path) -> None:
    cml = build_dir / "CMakeLists.txt"
    if cml.exists():
        text = cml.read_text()
        if "GIT_TAG" not in text:
            text = text.replace(
                "GIT_REPOSITORY https://github.com/pybind/pybind11.git)",
                f"GIT_REPOSITORY https://github.com/pybind/pybind11.git\n  GIT_TAG {PYBIND11_TAG})",
            )
        if "OSQP_MACOS_LIBCXX_FIX" not in text:
            text = text.replace(
                "pybind11_add_module(osqp_generated ${OSQP_SOURCES} ${EXT_SOURCES})\n"
                "target_link_libraries(osqp_generated PUBLIC pybind11::module osqpstatic)",
                "pybind11_add_module(osqp_generated ${OSQP_SOURCES} ${EXT_SOURCES})\n"
                "target_link_libraries(osqp_generated PUBLIC pybind11::module osqpstatic)\n"
                "\n"
                "# OSQP_MACOS_LIBCXX_FIX: some macOS Command Line Tools installs ship an incomplete\n"
                "# libc++ (missing <iostream>) on the default header search path even though the SDK\n"
                "# carries the full headers; point the compiler there too. Harmless when the default\n"
                "# toolchain headers are already complete (it's just an extra search path).\n"
                "if(APPLE)\n"
                "  execute_process(COMMAND xcrun --show-sdk-path\n"
                "                  OUTPUT_VARIABLE OSQP_MACOS_SDK_PATH\n"
                "                  OUTPUT_STRIP_TRAILING_WHITESPACE)\n"
                "  target_compile_options(osqp_generated PRIVATE\n"
                "    -isystem \"${OSQP_MACOS_SDK_PATH}/usr/include/c++/v1\")\n"
                "endif()\n",
            )
        cml.write_text(text)
    setup = build_dir / "setup.py"
    if setup.exists():
        text = setup.read_text()
        # Drop the hard-coded VS2022 generator; let CMake pick its platform default
        # (or honour a user-set CMAKE_GENERATOR env var).
        text = re.sub(r".*\['-G', 'Visual Studio 17 2022'\].*\n", "", text)
        setup.write_text(text)


class FastSolverBuildError(RuntimeError):
    """Raised when compiling the codegen'd OSQP extension fails."""


def build(system: str, N: int | None = None, smoke_test: bool = False,
          model: "RocketDirect | Rocket | Quadrotor | Pendulum | None" = None) -> None:
    out = codegen(system, N=N, smoke_test=smoke_test, model=model)
    print(f"[rnmpc] Wrote codegen sources to {out}")
    print(f"[rnmpc] Building + installing the fast solver for {system} (needs a C compiler + CMake) ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--use-pep517", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0:
        raise FastSolverBuildError(
            "Building the fast solver failed - a C/C++ compiler + CMake are required.\n"
            "  macOS: xcode-select --install\n"
            "         (still failing with \"'iostream' file not found\"? your Command Line Tools\n"
            "         are missing headers; run: sudo rm -rf /Library/Developer/CommandLineTools\n"
            "         && xcode-select --install, then retry)\n"
            "  Debian/Ubuntu: sudo apt install build-essential cmake\n"
            "  Windows: install the Visual Studio Build Tools (\"Desktop development with C++\") + CMake\n"
            "You can still run experiments without the fast solver: pass --stock-osqp.\n"
            "\n"
            "--- build output (tail) ---\n"
            + "\n".join(result.stdout.splitlines()[-40:])
        )
    print("[rnmpc] Fast solver installed.")
