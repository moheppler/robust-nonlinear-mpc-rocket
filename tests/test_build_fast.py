import pytest

import solver.build_fast as bf


def test_system_models_registry():
    assert set(bf.SYSTEM_MODELS) == {"pendulum", "quadrotor", "rocket", "rocket-compare"}


def test_codegen_writes_absolute_build_dir(tmp_path, monkeypatch):
    # keep the real repo build/ untouched: redirect CODEGEN_DIR
    monkeypatch.setattr(bf, "CODEGEN_DIR", tmp_path / "osqp_fast")
    out = bf.codegen("pendulum", N=3)
    assert out == tmp_path / "osqp_fast"
    assert (out / "setup.py").exists()
    assert (out / "CMakeLists.txt").exists()


def test_patch_pins_pybind11_and_relaxes_generator(tmp_path):
    cml = tmp_path / "CMakeLists.txt"
    cml.write_text(
        "include(FetchContent)\n"
        "FetchContent_Declare(\n  pybind11\n"
        "  GIT_REPOSITORY https://github.com/pybind/pybind11.git)\n"
        "FetchContent_MakeAvailable(pybind11)\n"
    )
    setup = tmp_path / "setup.py"
    setup.write_text("cmake_args += ['-G', 'Visual Studio 17 2022']\n")

    bf.patch_generated(tmp_path)

    assert "GIT_TAG" in cml.read_text()
    assert "v2.13.6" in cml.read_text()
    assert "Visual Studio 17 2022" not in setup.read_text()


def test_patch_injects_macos_libcxx_fix(tmp_path):
    cml = tmp_path / "CMakeLists.txt"
    cml.write_text(
        "pybind11_add_module(osqp_generated ${OSQP_SOURCES} ${EXT_SOURCES})\n"
        "target_link_libraries(osqp_generated PUBLIC pybind11::module osqpstatic)"
    )
    bf.patch_generated(tmp_path)
    text = cml.read_text()
    assert "OSQP_MACOS_LIBCXX_FIX" in text
    assert "if(APPLE)" in text
    assert "xcrun --show-sdk-path" in text

    # idempotent: patching an already-patched file doesn't duplicate the block
    bf.patch_generated(tmp_path)
    assert cml.read_text().count("if(APPLE)") == 1


@pytest.mark.parametrize("smoke_test,N,expected", [
    (True, None, 3),
    (False, None, 15),
    (False, 7, 7),
])
def test_codegen_smoke_test_and_N_selection(tmp_path, monkeypatch, smoke_test, N, expected):
    monkeypatch.setattr(bf, "CODEGEN_DIR", tmp_path / "osqp_fast")
    captured = {}
    real_qp = bf.QP

    def spy_qp(N_arg, *a, **k):
        captured["N"] = N_arg
        return real_qp(N_arg, *a, **k)

    monkeypatch.setattr(bf, "QP", spy_qp)
    bf.codegen("pendulum", N=N, smoke_test=smoke_test)
    assert captured["N"] == expected


def test_build_raises_clean_error_on_compile_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "CODEGEN_DIR", tmp_path / "osqp_fast")
    monkeypatch.setattr(bf, "codegen", lambda system, N=None, smoke_test=False: tmp_path / "osqp_fast")

    class _FailedResult:
        returncode = 1
        stdout = "fatal error: 'iostream' file not found\n"

    monkeypatch.setattr(bf.subprocess, "run", lambda *a, **k: _FailedResult())
    with pytest.raises(bf.FastSolverBuildError, match="iostream"):
        bf.build("pendulum")
