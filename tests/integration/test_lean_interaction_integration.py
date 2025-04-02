# File: tests/integration/test_lean_interaction_integration.py

"""Integration tests for the lean_interaction module.

These tests verify the functionality of `check_and_compile_item`, focusing on
its interaction with the Lean ecosystem (via the `lake` executable) and the
Knowledge Base database. Tests cover successful compilation with and without
dependencies, caching behavior, various failure modes (syntax, proof, missing
dependencies, configuration errors), and status updates in the database.

Requires the 'lake' executable to be available in the system PATH or specified
via the LAKE_EXECUTABLE environment variable. Requires necessary Lean libraries
(like the persistent shared library path) to be configured via environment variables
as used by `lean_interaction.py`.
"""

import pytest
import os
import sys
import pathlib
import shutil
import time
import asyncio
import subprocess
import logging # Added logging
from unittest.mock import AsyncMock, MagicMock, call # Keep mocks if needed for specific tests
from dotenv import load_dotenv; load_dotenv()

# Adjust path to import from src
# (Assuming project structure allows this)
try:
    from lean_automator.kb_storage import (
        initialize_database,
        save_kb_item,
        get_kb_item_by_name,
        KBItem,
        ItemStatus,
        ItemType
    )
    from lean_automator.lean_interaction import check_and_compile_item
except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr)
    raise

# --- Logger ---
logger = logging.getLogger(__name__)

# --- Configuration ---
LAKE_EXEC_PATH = os.environ.get('LAKE_EXECUTABLE', 'lake')
# Note: lean_interaction.py now hardcodes the temp lib name within its helper.
# TEMP_LIB_NAME is no longer passed to check_and_compile_item.
# Keep it here if needed for direct path construction in tests, otherwise remove.
# TEMP_LIB_NAME = "TestIntegrationLib"
CACHE_ENV_VAR = 'LEAN_AUTOMATOR_LAKE_CACHE' # Environment variable name expected by lean_interaction

def is_lake_available(lake_exec=LAKE_EXEC_PATH):
    """Checks if the specified lake executable is found."""
    return shutil.which(lake_exec) is not None

# --- Apply marks ---
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not is_lake_available(), reason=f"Lake executable '{LAKE_EXEC_PATH}' not found in PATH or via LAKE_EXECUTABLE env var.")
]

# --- Fixtures ---

@pytest.fixture
def test_db(tmp_path):
    """Provides a temporary, initialized SQLite database for each test.

    Creates an empty SQLite file within the pytest temporary directory (`tmp_path`)
    and initializes the schema using `kb_storage.initialize_database`.

    Args:
        tmp_path: The pytest fixture providing a temporary directory unique to each test function.

    Yields:
        str: The absolute path string to the temporary database file.
    """
    db_file = tmp_path / "test_integration_lean.sqlite"
    db_path = str(db_file)
    logger.debug(f"Creating test database: {db_path}")
    initialize_database(db_path=db_path)
    yield db_path
    # tmp_path fixture handles cleanup of the directory and file

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provides a temporary directory path intended for use as LAKE_HOME.

    Creates a subdirectory named "lake_cache" within the pytest temporary
    directory (`tmp_path`). This path can be set as the `LAKE_HOME` environment
    variable (e.g., via `monkeypatch.setenv`) to test Lake's caching behavior
    in an isolated manner.

    Args:
        tmp_path: The pytest fixture providing a temporary directory unique to each test function.

    Yields:
        str: The absolute path string to the temporary cache directory.
    """
    cache_dir = tmp_path / "lake_cache"
    logger.debug(f"Creating temporary Lake cache directory: {cache_dir}")
    cache_dir.mkdir()
    yield str(cache_dir)
    # tmp_path fixture handles cleanup of the directory

# --- Test Cases (Updated) ---

@pytest.mark.asyncio
async def test_compile_success_no_deps_no_cache(test_db, monkeypatch):
    """Verify successful compilation of a simple item with no dependencies and no cache."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False) # Ensure cache var is NOT set
    item_name = "Test.NoDeps.NoCache"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def simpleDef : Nat := 5", topic="Test",
        latex_statement="Dummy", # Required field
        status=ItemStatus.PENDING_LEAN # Set appropriate initial status
    )
    await save_kb_item(item, client=None, db_path=test_db)

    # Call the function under test
    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
        # temp_lib_name removed, handled internally
    )

    # Assertions
    assert success is True, f"Compilation should succeed, but failed: {message}"
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None, "Item not found after successful compilation"
    assert retrieved.status == ItemStatus.PROVEN, "Item status should be PROVEN after success"


@pytest.mark.asyncio
async def test_compile_success_no_deps_with_cache(test_db, temp_cache_dir, monkeypatch):
    """Verify successful compilation of a simple item using Lake caching."""
    monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir) # Set cache var via monkeypatch
    item_name = "Test.NoDeps.WithCache"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def simpleDefWC : Nat := 6", topic="Test",
        latex_statement="Dummy", status=ItemStatus.PENDING_LEAN
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    assert success is True, f"Compilation should succeed with cache, but failed: {message}"
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None, "Item not found after successful compilation"
    assert retrieved.status == ItemStatus.PROVEN, "Item status should be PROVEN"
    # Note: Verifying specific files *inside* the LAKE_HOME cache can be brittle.
    # It's generally sufficient to test that enabling the cache doesn't break compilation
    # and potentially measure timing differences (though timing can be flaky).


@pytest.mark.asyncio
@pytest.mark.skip(reason="Test logic relies on older temp env setup, needs update for shared lib")
async def test_compile_success_with_deps_cached(test_db, temp_cache_dir, monkeypatch):
    """Verify compiling an item succeeds when its dependency is already built/cached. (Skipped - Needs Update)"""
    # NOTE: This test assumed the old temporary environment setup where dependencies
    # were copied into the temp lib. With the shared library approach, Lake resolves
    # dependencies differently. The core idea (reusing builds) still applies via
    # Lake's mechanisms (potentially via LAKE_HOME cache or shared build outputs),
    # but the test setup/assertions need revision for the shared library model.
    pytest.skip("Test needs refactoring for shared library dependency model.")

    # --- Original Logic (for reference) ---
    # monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir) # Use cache
    # dep_name = "Test.DepA.Cached"
    # dep_item = KBItem(unique_name=dep_name, item_type=ItemType.DEFINITION, lean_code="def depValueCached : Nat := 10", topic="Test", latex_statement="Dummy", status=ItemStatus.PENDING_LEAN)
    # await save_kb_item(dep_item, client=None, db_path=test_db)
    # dep_success, _ = await check_and_compile_item(dep_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH)
    # assert dep_success is True
    #
    # target_name = "Test.TargetB.UsesCached"
    # target_code = f"import Test.DepA.Cached\n\ntheorem useDepCached : depValueCached = 10 := rfl" # Import adjusted
    # target_item = KBItem(unique_name=target_name, item_type=ItemType.THEOREM, lean_code=target_code, topic="Test", latex_statement="Dummy", latex_proof="Dummy", plan_dependencies=[dep_name], status=ItemStatus.PENDING_LEAN)
    # await save_kb_item(target_item, client=None, db_path=test_db)
    #
    # start_time = time.monotonic()
    # target_success, message = await check_and_compile_item(target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH)
    # end_time = time.monotonic()
    # logger.info(f"Target build time (cached dep): {end_time - start_time:.4f}s")
    #
    # assert target_success is True, f"Compilation failed: {message}"
    # retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    # assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True])
async def test_compile_fail_syntax_error(test_db, temp_cache_dir, monkeypatch, use_cache):
    """Verify compilation fails correctly for Lean code with syntax errors."""
    if use_cache: monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else: monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"Test.SyntaxErr.{'Cache' if use_cache else 'NoCache'}"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def oops : Nat :=", topic="Test", latex_statement="Dummy",
        status=ItemStatus.PENDING_LEAN
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    assert success is False, "Compilation should fail for syntax error"
    assert "Lean validation failed" in message, "Failure message mismatch"
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None, "Item not found after failed compilation"
    assert retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED, "Status should be LEAN_VALIDATION_FAILED"
    assert retrieved.lean_error_log is not None and "error:" in retrieved.lean_error_log, "Error log should contain 'error:'"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True])
async def test_compile_fail_proof_error(test_db, temp_cache_dir, monkeypatch, use_cache):
    """Verify compilation fails correctly for Lean code with proof errors."""
    if use_cache: monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else: monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"Test.ProofErr.{'Cache' if use_cache else 'NoCache'}"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.THEOREM,
        lean_code="theorem badProof : 1 + 1 = 3 := rfl", topic="Test",
        latex_statement="Dummy", latex_proof="Dummy", status=ItemStatus.PENDING_LEAN
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    assert success is False, "Compilation should fail for proof error"
    assert "Lean validation failed" in message, "Failure message mismatch"
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None, "Item not found after failed compilation"
    assert retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED, "Status should be LEAN_VALIDATION_FAILED"
    assert retrieved.lean_error_log is not None and "error:" in retrieved.lean_error_log.lower(), "Error log should contain 'error:'"
    assert "failed to synthesize instance" in retrieved.lean_error_log.lower() or "motive is not type correct" in retrieved.lean_error_log.lower() , "Error log content mismatch for proof error"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Test logic assumes missing dependency is fatal early; shared lib requires build check.")
async def test_compile_fail_missing_dependency_db(test_db, temp_cache_dir, monkeypatch, use_cache):
    """Verify failure when a listed plan_dependency is missing from the DB. (Skipped - Needs Update)"""
    # NOTE: The shared library model changes *when* this error occurs.
    # The old model failed early during dependency fetching.
    # The new model will likely fail during the `lake build` command in the temp env
    # because the `import Missing.Dep` will fail.
    # This test needs to be adapted to check for the expected `lake build` failure message.
    pytest.skip("Test needs refactoring for shared library dependency model failure mode.")

    # --- Original Logic (for reference) ---
    # if use_cache: monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    # else: monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    #
    # item_name = f"Test.MissingDepDB.{'Cache' if use_cache else 'NoCache'}"
    # non_existent_dep = "Test.DoesNotExist"
    # lean_code = f"import Test.DoesNotExist\n\ntheorem usesMissing : True := trivial" # Import adjusted
    # item = KBItem(unique_name=item_name, item_type=ItemType.THEOREM, lean_code=lean_code, topic="Test", latex_statement="Dummy", latex_proof="Dummy", plan_dependencies=[non_existent_dep], status=ItemStatus.PENDING_LEAN)
    # await save_kb_item(item, client=None, db_path=test_db)
    #
    # success, message = await check_and_compile_item(item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH)
    #
    # assert success is False
    # # Original assertion checked for early DB fetch error, now expect Lake build error
    # # assert "Dependency error" in message
    # # assert f"Item '{non_existent_dep}' not found" in message
    # assert "Lean validation failed" in message # Expect build failure
    # retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    # assert retrieved is not None and retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED # Expect validation fail status
    # assert retrieved.lean_error_log is not None
    # assert f"unknown package '{non_existent_dep.split('.')[0]}'" in retrieved.lean_error_log or f"unknown module '{non_existent_dep}'" in retrieved.lean_error_log # Check for Lake error


@pytest.mark.asyncio
async def test_compile_success_dep_proven_in_shared_lib(test_db, monkeypatch):
    """Verify successful compilation when dependency is PROVEN (assumed in shared lib)."""
    # Note: This test implicitly assumes the dependency is already correctly built
    # in the shared library referenced by the environment setup.
    # We don't interact with the shared library directly here, only ensure the
    # target item *references* a dependency assumed to be valid there.
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False) # Cache less relevant here
    dep_name = "Test.SharedDep.Exists" # Name of dependency assumed to be in shared lib
    # Save a dummy item for the dependency just so the DB lookup for plan_dependencies doesn't fail early.
    # Its actual code isn't used in the temp build env. Status must be PROVEN.
    dep_item = KBItem(unique_name=dep_name, item_type=ItemType.DEFINITION, lean_code="# Proven in Shared Lib", status=ItemStatus.PROVEN, latex_statement="Dummy")
    await save_kb_item(dep_item, client=None, db_path=test_db)

    target_name = "Test.TargetUsesSharedDep.NoCache"
    # Lean code imports the dependency by its *real* name (not temp lib prefix)
    target_code = f"import {dep_name}\n\ntheorem useSharedDep : True := trivial -- Placeholder"
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM,
        lean_code=target_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[dep_name], # List the dependency
        status=ItemStatus.PENDING_LEAN
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    # Expect success because the temp lakefile requires the shared lib where the dep lives
    assert success is True, f"Compilation failed using shared library dependency: {message}"
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN


# --- Configuration Error Tests ---

@pytest.mark.asyncio
async def test_compile_fail_lake_not_found(test_db):
    """Verify failure when the specified Lake executable path is invalid."""
    item_name = "Test.LakeNotFound.Config"
    item = KBItem(unique_name=item_name, lean_code="def x := 1", latex_statement="Dummy", status=ItemStatus.PENDING_LEAN)
    await save_kb_item(item, client=None, db_path=test_db)

    invalid_lake_path = "/path/to/nonexistent/lake_executable_qwerty_int"
    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=invalid_lake_path
    )

    assert success is False, "Compilation should fail if lake executable is invalid"
    assert "Lake executable not found" in message or "failed" in message.lower(), "Failure message mismatch for invalid lake path"
    if "Lake executable not found" in message: # Check only if specific message present
        assert invalid_lake_path in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    # Status should remain unchanged or become ERROR depending on where the failure is caught
    assert retrieved is not None
    assert retrieved.status != ItemStatus.PROVEN, "Status should not be PROVEN on config error"


# --- Timeout Test (Mocked) ---

@pytest.mark.asyncio
async def test_compile_fail_timeout_mocked(test_db, mocker):
    """Verify failure handling when the 'lake build' subprocess times out (mocked)."""
    item_name = "Test.TimeoutMocked.Int"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def anyContent := 1", topic="Test", latex_statement="Dummy",
        status=ItemStatus.PENDING_LEAN
    )
    await save_kb_item(item, client=None, db_path=test_db)

    short_timeout = 5 # Timeout value for the test
    # Mock subprocess.run to raise TimeoutExpired specifically for 'lake build'
    mock_timeout_exception = subprocess.TimeoutExpired(cmd=["mock", "lake", "build"], timeout=short_timeout)

    def mock_run_side_effect(*args, **kwargs):
        cmd_list = kwargs.get('args', args[0] if args else [])
        # Only raise timeout for the 'lake build' command during verification
        # Need to identify the build command more robustly if possible
        # Check if 'build' is in the command and it's likely the temp verification one
        if isinstance(cmd_list, list) and lake_executable_path in cmd_list[0] and 'build' in cmd_list:
             # Rough check: is the target module name likely the temp one?
             # This is fragile. A better way might be needed if other builds happen.
             if any(name_part in cmd_list[-1] for name_part in ["TempVerifyLib", item_name]):
                 logger.debug(f"Mocking TimeoutExpired for command: {cmd_list}")
                 raise mock_timeout_exception
        # Allow other calls (like lean --print-libdir) to proceed normally (or mock success)
        logger.debug(f"Mock allowing non-build command: {cmd_list}")
        # Simulate success for other potential calls like stdlib detection
        mock_success_result = MagicMock(spec=subprocess.CompletedProcess, stdout="/mock/lean/lib", returncode=0)
        return mock_success_result

    mocker.patch('subprocess.run', side_effect=mock_run_side_effect)
    # Mock shutil.which if needed
    mocker.patch('shutil.which', return_value=LAKE_EXEC_PATH) # Ensure lake is "found"


    # Call the function under test
    success, message = await check_and_compile_item(
         item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
         timeout_seconds=short_timeout
    )

    # Assertions
    assert success is False, "Compilation should fail on timeout"
    assert "Timeout" in message or "timed out" in message.lower(), "Failure message should indicate timeout"
    assert f"{short_timeout}s" in message, "Failure message should mention timeout duration"

    # Assert DB state shows ERROR and timeout message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None, "Item not found after timeout"
    assert retrieved.status == ItemStatus.ERROR, "Status should be ERROR after timeout"
    assert retrieved.lean_error_log is not None, "Error log should be populated after timeout"
    assert "Timeout" in retrieved.lean_error_log or "timed out" in retrieved.lean_error_log.lower(), "Error log content mismatch for timeout"
    assert f"{short_timeout}s" in retrieved.lean_error_log, "Error log should mention timeout duration"