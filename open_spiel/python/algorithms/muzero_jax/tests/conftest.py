import os
import jax
import pytest
import shutil
import time
import wandb
import gc
from _pytest._code.code import ExceptionInfo

os.environ["JAX_DISABLE_JIT"] = "1"   # must precede the first JAX import
jax.config.update("jax_disable_jit", True) 

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to prevent memory leaks by removing exception info after test completion."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

    # Remove exception info, since it causes excessive memory usage and workers crash
    if call.excinfo and isinstance(call.excinfo, ExceptionInfo):
        call.excinfo = None

@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global state before and after each test to prevent state pollution."""
    # Before test: Clean up any existing WandB runs
    if wandb.run is not None:
        try:
            wandb.finish()
        except:
            pass  # Ignore errors during cleanup
    
    yield  # Run the test
    
    # After test: Clean up WandB state and force garbage collection
    if wandb.run is not None:
        try:
            wandb.finish()
        except:
            pass  # Ignore errors during cleanup
    
    # Force garbage collection to free memory
    gc.collect()

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Clear JAX caches before each test to manage memory usage."""
    # Clear JAX caches every 5 tests to prevent excessive memory usage (more frequent for integration tests)
    if hasattr(item.session, '_test_count'):
        item.session._test_count += 1
    else:
        item.session._test_count = 1
    
    if item.session._test_count % 5 == 0:
        try:
            jax.clear_caches()
            gc.collect()
        except:
            pass

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test to prevent memory leaks."""
    # Force garbage collection after each test
    gc.collect()
    
    # Clear JAX caches every 10 tests to manage memory (more frequent for integration tests)
    if hasattr(item.session, '_test_count') and item.session._test_count % 10 == 0:
        try:
            jax.clear_caches()
            gc.collect()
        except:
            pass 

# ---------------------------------------------------------------------------
# WandB console capture monkey-patch
# ---------------------------------------------------------------------------

# WandB's console capture can raise `ValueError: I/O operation on closed file`
# during test teardown when streams are already closed.  This propagates to
# the logging system and causes the pytest process to exit with a non-zero
# return code, which the coverage runner interprets as a test failure even
# though all assertions passed.  We patch the offending function to swallow
# the specific error and keep the process exit code at 0.

try:
    import wandb  # noqa: WPS433 – third-party import inside try
    from wandb.sdk.lib import console_capture as _cc  # noqa: WPS433

    _orig_write_with_callbacks = _cc.write_with_callbacks  # type: ignore[attr-defined]

    def _safe_write_with_callbacks(s: str, *args, **kwargs):  # type: ignore[override]
        """Wrapper that ignores `ValueError` from closed streams.

        WandB sometimes attempts to write to stdout/stderr after those streams
        are already closed (especially in short-lived subprocesses).  The
        resulting `ValueError` bubbles through `logging` and flips the process
        exit code to non-zero.  We swallow that *specific* error so the test
        process ends cleanly.
        """

        try:
            return _orig_write_with_callbacks(s, *args, **kwargs)
        except ValueError:
            return len(s)

    _cc.write_with_callbacks = _safe_write_with_callbacks  # type: ignore[assignment]
except Exception:  # pragma: no cover – patching is best-effort
    pass  # If WandB import fails, nothing to patch 