"""``rnmpc`` - a single entry point for the robust nonlinear MPC experiments.

Examples:
    rnmpc pendulum --run           # run the closed-loop experiment (generate data)
    rnmpc pendulum                 # plot the most recent result
    rnmpc rocket --run --smoke-test
    rnmpc build-fast pendulum      # (re)build the compiled fast solver for a system
"""
import argparse
import importlib
import sys

import solver.osqp_backend as ob

SYSTEMS = {
    "pendulum": "expe.main_pendulum_robust_closed_loop",
    "quadrotor": "expe.main_quadrotor_robust_closed_loop",
    "rocket": "expe.main_rocket_robust_closed_loop",
    "rocket-compare": "expe.main_rocket_compare_closed_loop",
    "rocket-direct": "expe.main_rocket_direct_closed_loop",
}


def _dispatch_experiment(system: str, run: bool, N, smoke_test: bool) -> int:
    mod = importlib.import_module(SYSTEMS[system])
    if run:
        mod.generate(N=N, smoke_test=smoke_test)
    else:
        mod.plot()
    return 0


def _dispatch_build(system: str, N, smoke_test: bool) -> int:
    from solver.build_fast import build, FastSolverBuildError
    try:
        build(system, N=N, smoke_test=smoke_test)
    except FastSolverBuildError as e:
        print(f"[rnmpc] {e}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="rnmpc",
        description="Run robust nonlinear MPC (SLS tube MPC) closed-loop experiments.",
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build-fast", help="Codegen + compile the fast OSQP solver for a system.")
    build.add_argument("system", choices=sorted(SYSTEMS))
    build.add_argument("--N", type=int, default=None, help="Horizon length.")
    build.add_argument("--smoke-test", action="store_true",
                        help="Build for N=3 to match `rnmpc <system> --run --smoke-test` "
                             "(default build is N=15, matching a full run).")

    for name in sorted(SYSTEMS):
        p = sub.add_parser(name, help=f"Run/plot the {name} experiment.")
        p.add_argument("--run", action="store_true", help="Run the simulation (generate data).")
        p.add_argument("--N", type=int, default=None, help="Horizon length (overrides the default).")
        p.add_argument("--smoke-test", action="store_true",
                       help="Short, cheap variant to validate the pipeline end-to-end.")
        p.add_argument("--stock-osqp", action="store_true",
                       help="Force the pure-Python OSQP backend (no compiler needed, slower).")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "build-fast":
        return _dispatch_build(args.system, args.N, args.smoke_test)

    if getattr(args, "stock_osqp", False):
        ob.set_force_stock(True)
    try:
        return _dispatch_experiment(args.command, args.run, args.N, args.smoke_test)
    except ob.FastSolverRequiredError as e:
        print(f"[rnmpc] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
