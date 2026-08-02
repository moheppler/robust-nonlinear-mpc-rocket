# Verification workflow for robust-nonlinear-mpc.
#
#   make check      - fast: syntax, import, lint (if available), and CLI validation. No codegen build required.
#   make build       - build the codegen'd OSQP solver for the pendulum system (one-time, cached).
#   make smoke      - build (if needed) + a short, cheap run of the pendulum closed-loop experiment.
#   make experiment - build (if needed) + the full pendulum closed-loop experiment (expensive: do not run
#                     automatically; only on explicit request).

PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

empty :=
space := $(empty) $(empty)
comma := ,

PY_SRC := $(shell find dyn solver util expe -name '*.py')

# Modules that import cleanly without the codegen'd OSQP extension (build/osqp_fast).
CORE_MODULES := dyn.model dyn.pendulum dyn.quadrotor dyn.rocket dyn.LTI dyn.LTV dyn.LTI_OF dyn.LTV_OF \
                 util.SLS util.plot util.footnote util.quaternion_to_euler util.euler_to_quaternion util.rectangle_coordinates \
                 solver.ocp solver.nlp solver.nlp_soft_constraints solver.qp_jit

BUILD_MARKER := build/osqp_fast/.built

.PHONY: check build smoke experiment clean-build

check:
	@echo "== [1/4] Syntax check =="
	$(PYTHON) -m py_compile $(PY_SRC)
	@echo "== [2/4] Import check (modules that don't require the codegen'd solver) =="
	$(PYTHON) -c "import $(subst $(space),$(comma) ,$(CORE_MODULES))" > /dev/null
	@echo "== [3/4] Lint =="
	@if command -v ruff > /dev/null 2>&1; then \
		ruff check dyn solver util expe ; \
	elif command -v flake8 > /dev/null 2>&1; then \
		flake8 dyn solver util expe ; \
	else \
		echo "no linter (ruff/flake8) installed in this environment; skipping lint step" ; \
	fi
	@echo "== [4/4] CLI validation (argparse only, no solve) =="
	$(PYTHON) expe/main_pendulum_robust_closed_loop.py --help > /dev/null
	@echo "check: OK"

$(BUILD_MARKER):
	$(PYTHON) -c "from solver.build_fast import build; build('pendulum')"
	touch $(BUILD_MARKER)

build: $(BUILD_MARKER)

smoke: build
	$(PYTHON) expe/main_pendulum_robust_closed_loop.py --run --smoke-test

experiment: build
	$(PYTHON) expe/main_pendulum_robust_closed_loop.py --run

clean-build:
	rm -rf build
