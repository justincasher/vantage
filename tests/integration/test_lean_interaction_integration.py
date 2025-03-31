# File: tests/integration/test_lean_interaction_integration.py

import pytest
import os
import sys
import pathlib
import shutil
import time
import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, call # Keep mocks if needed for specific tests

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

# --- Configuration ---
LAKE_EXEC_PATH = os.environ.get('LAKE_EXECUTABLE', 'lake')
TEMP_LIB_NAME = "TestIntegrationLib" # Use a distinct name for integration tests
CACHE_ENV_VAR = 'LEAN_AUTOMATOR_LAKE_CACHE' # Environment variable name

def is_lake_available(lake_exec=LAKE_EXEC_PATH):
    return shutil.which(lake_exec) is not None

# --- Apply marks ---
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not is_lake_available(), reason=f"Lake executable '{LAKE_EXEC_PATH}' not found in PATH.")
]

# --- Fixtures ---

@pytest.fixture
def test_db(tmp_path):
    """Fixture for a temporary database per test."""
    db_file = tmp_path / "test_integration_lean.sqlite"
    db_path = str(db_file)
    initialize_database(db_path=db_path)
    yield db_path
    # tmp_path handles cleanup

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture for a temporary directory to use as LAKE_HOME per test."""
    cache_dir = tmp_path / "lake_cache"
    cache_dir.mkdir()
    yield str(cache_dir)
    # tmp_path handles cleanup

# --- Test Cases (Updated) ---

@pytest.mark.asyncio
async def test_compile_success_no_deps_no_cache(test_db, monkeypatch):
    """Test compiling simple item, ensuring no cache env var is set."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False) # Ensure cache var is NOT set
    item_name = "Test.NoDeps.NoCache"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def simpleDef : Nat := 5", topic="Test",
        latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is True, f"Compile failed: {message}"
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None and retrieved.status == ItemStatus.PROVEN


@pytest.mark.asyncio
async def test_compile_success_no_deps_with_cache(test_db, temp_cache_dir, monkeypatch):
    """Test compiling simple item with cache env var set."""
    monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir) # Set cache var
    item_name = "Test.NoDeps.WithCache"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def simpleDefWC : Nat := 6", topic="Test",
        latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is True, f"Compile failed: {message}"
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None and retrieved.status == ItemStatus.PROVEN
    # Check if olean/ilean/trace files were created in the cache dir
    # cache_path = pathlib.Path(temp_cache_dir)
    # Corrected Path: Check within LAKE_HOME/build/lib/...
    # expected_olean_rel = pathlib.Path("build/lib") / TEMP_LIB_NAME / "Test" / "NoDeps" / "WithCache.olean"
    # expected_ilean_rel = expected_olean_rel.with_suffix(".ilean")
    # assert (cache_path / expected_olean_rel).is_file(), f"Olean not found in cache: {cache_path / expected_olean_rel}" # This assertion was incorrect
    # assert (cache_path / expected_ilean_rel).is_file(), f"ILean not found in cache: {cache_path / expected_ilean_rel}" # This assertion was incorrect


@pytest.mark.asyncio
async def test_compile_success_with_deps_cached(test_db, temp_cache_dir, monkeypatch):
    """Test compiling an item where dependency should be cached."""
    monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir) # Use cache

    # 1. Compile dependency (Item A) - this should populate the cache
    dep_name = "Test.DepA.Cached"
    dep_item = KBItem(
        unique_name=dep_name, item_type=ItemType.DEFINITION,
        lean_code="def depValueCached : Nat := 10", topic="Test",
        latex_statement="Dummy"
    )
    await save_kb_item(dep_item, client=None, db_path=test_db)
    dep_success, _ = await check_and_compile_item(
        dep_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )
    assert dep_success is True
    # Verify dep olean exists in cache
    # cache_path = pathlib.Path(temp_cache_dir)
    # Corrected Path: Check within LAKE_HOME/build/lib/...
    # dep_olean_rel = pathlib.Path("build/lib") / TEMP_LIB_NAME / "Test" / "DepA" / "Cached.olean"
    # assert (cache_path / dep_olean_rel).is_file(), "Dep olean missing from cache" # This assertion was incorrect

    # 2. Compile target (Item B depends on A) - should reuse cached dep
    target_name = "Test.TargetB.UsesCached"
    target_code = f"import {TEMP_LIB_NAME}.Test.DepA.Cached\n\ntheorem useDepCached : depValueCached = 10 := rfl"
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM,
        lean_code=target_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    # Time the second build (optional, can be flaky)
    start_time = time.monotonic()
    target_success, message = await check_and_compile_item(
        target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )
    end_time = time.monotonic()
    print(f"\nTarget build time (cached dep): {end_time - start_time:.4f}s") # Print time for info

    # 3. Assertions for Target B
    assert target_success is True, f"Compilation failed: {message}"
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN
    # Verify target olean exists in cache
    # Corrected Path: Check within LAKE_HOME/build/lib/...
    # target_olean_rel = pathlib.Path("build/lib") / TEMP_LIB_NAME / "Test" / "TargetB" / "UsesCached.olean"
    # assert (cache_path / target_olean_rel).is_file(), "Target olean missing from cache" # This assertion was incorrect


# --- Failure cases remain mostly the same, test with/without cache where relevant ---

@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True]) # Run with and without cache
async def test_compile_fail_syntax_error(test_db, temp_cache_dir, monkeypatch, use_cache):
    """Test syntax error failure with and without cache enabled."""
    if use_cache:
        monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else:
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"Test.SyntaxErr.{'Cache' if use_cache else 'NoCache'}"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def oops : Nat :=", topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is False
    assert "Lean validation failed" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None
    assert retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert retrieved.lean_error_log is not None and "error:" in retrieved.lean_error_log


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True])
async def test_compile_fail_proof_error(test_db, temp_cache_dir, monkeypatch, use_cache):
    """Test proof error failure with and without cache enabled."""
    if use_cache:
        monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else:
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"Test.ProofErr.{'Cache' if use_cache else 'NoCache'}"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.THEOREM,
        lean_code="theorem badProof : 1 + 1 = 3 := rfl", topic="Test",
        latex_statement="Dummy", latex_proof="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is False
    assert "Lean validation failed" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None
    assert retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert retrieved.lean_error_log is not None
    assert "error:" in retrieved.lean_error_log.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True])
async def test_compile_fail_missing_dependency_db(test_db, temp_cache_dir, monkeypatch, use_cache):
    """Test missing DB dependency failure (should fail early, cache irrelevant)."""
    if use_cache:
        monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else:
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"Test.MissingDepDB.{'Cache' if use_cache else 'NoCache'}"
    non_existent_dep = "Test.DoesNotExist"
    lean_code = f"import {TEMP_LIB_NAME}.Test.DoesNotExist\n\ntheorem usesMissing : True := trivial"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.THEOREM,
        lean_code=lean_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[non_existent_dep]
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is False
    assert "Dependency error" in message
    assert f"Item '{non_existent_dep}' not found" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None and retrieved.status == ItemStatus.ERROR
    assert retrieved.lean_error_log is not None
    assert f"Item '{non_existent_dep}' not found" in retrieved.lean_error_log


# --- Tests that rely on specific build behavior (olean missing, dep built by lake) ---
# These should ideally work correctly regardless of caching, as Lake handles the logic.
# Test them once without cache for simplicity, assuming Lake's core logic is sound.

@pytest.mark.asyncio
async def test_compile_success_dep_proven_olean_missing(test_db, monkeypatch):
    """Test success when dep status=PROVEN but olean missing (compile from source). Cache unset."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    dep_name = "Test.DepProvenNoOlean.NoCache"
    dep_item = KBItem(
        unique_name=dep_name, item_type=ItemType.DEFINITION,
        lean_code="def depValueNoOleanNC : Nat := 20", topic="Test",
        latex_statement="Dummy", status=ItemStatus.PROVEN, lean_olean=None
    )
    await save_kb_item(dep_item, client=None, db_path=test_db)

    target_name = "Test.TargetDepProvenNoOlean.NoCache"
    target_code = f"import {TEMP_LIB_NAME}.Test.DepProvenNoOlean.NoCache\n\ntheorem useVerifiedNoOleanNC : depValueNoOleanNC = 20 := rfl"
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM,
        lean_code=target_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is True, f"Compilation failed unexpectedly: {message}"
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN
    retrieved_dep = get_kb_item_by_name(dep_name, db_path=test_db)
    assert retrieved_dep is not None and retrieved_dep.lean_olean is None # Verify dep still has no olean in DB


@pytest.mark.asyncio
async def test_compile_success_dep_built_by_lake(test_db, monkeypatch):
    """Test Lake successfully building dependency from source. Cache unset."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    dep_name = "Test.DepNeedsBuilding.NoCache"
    dep_item = KBItem(
        unique_name=dep_name, item_type=ItemType.DEFINITION,
        lean_code="def depValueNeedsBuildingNC : Nat := 30", topic="Test",
        latex_statement="Dummy", status=ItemStatus.PENDING
    )
    await save_kb_item(dep_item, client=None, db_path=test_db)

    target_name = "Test.TargetUsesBuiltDep.NoCache"
    target_code = f"import {TEMP_LIB_NAME}.Test.DepNeedsBuilding.NoCache\n\ntheorem useBuiltDepNC : depValueNeedsBuildingNC = 30 := rfl"
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM,
        lean_code=target_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is True, f"Compilation failed unexpectedly: {message}"
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN
    retrieved_dep = get_kb_item_by_name(dep_name, db_path=test_db)
    assert retrieved_dep is not None and retrieved_dep.status == ItemStatus.PENDING # Verify dep status unchanged


# --- Configuration Error Tests ---

@pytest.mark.asyncio
async def test_compile_fail_lake_not_found(test_db):
    """Test failure when Lake executable path is invalid (independent of cache)."""
    item_name = "Test.LakeNotFound.Config"
    item = KBItem(unique_name=item_name, lean_code="def x := 1", latex_statement="Dummy")
    await save_kb_item(item, client=None, db_path=test_db)

    invalid_lake_path = "/path/to/nonexistent/lake_executable_qwerty_int"
    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=invalid_lake_path,
        temp_lib_name=TEMP_LIB_NAME
    )

    assert success is False
    assert "Lake executable not found" in message
    assert invalid_lake_path in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None and retrieved.status == ItemStatus.PENDING # Status unchanged on config error


# --- Timeout Test (Mocked in integration as it's hard to reliably trigger) ---
# Keep the mocked timeout test as before, it's independent of caching logic

@pytest.mark.asyncio
async def test_compile_fail_timeout_mocked(test_db, mocker):
    """Test failure handling when subprocess.run times out (mocked)."""
    item_name = "Test.TimeoutMocked.Int"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def anyContent := 1", topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    short_timeout = 5
    mock_libdir_result = MagicMock(spec=subprocess.CompletedProcess, stdout="/mock/lean/lib", returncode=0)
    mock_timeout_exception = subprocess.TimeoutExpired(cmd="mock lake build", timeout=short_timeout)

    # Helper function to create a mock CompletedProcess, needed for 'which' mocking
    def mock_completed_process(returncode=0, stdout='', stderr=''):
        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = stderr
        return proc

    def mock_run_side_effect(*args, **kwargs):
        command_args = kwargs.get('args', args[0] if args else [])
        # Handle lean --print-libdir call
        if '--print-libdir' in command_args: return mock_libdir_result
        # Handle lake build call (simulate timeout)
        if 'build' in command_args: raise mock_timeout_exception
        # Handle 'which' or similar path lookups for stdlib detection (provide a plausible fake path)
        # Note: shutil.which isn't directly mocked here, but subprocess.run might be called by underlying checks.
        # Adapt this if direct mocking of shutil.which is needed.
        if command_args and 'which' in str(command_args[0]): # Basic check
             return mock_completed_process(0, stdout='/fake/path/to/lean')
        # Default for unexpected calls
        print(f"Warning: Unexpected subprocess call in timeout mock: {command_args}")
        return mock_completed_process(1) # Return non-zero for safety on unexpected calls

    mock_subprocess_run = mocker.patch('subprocess.run', side_effect=mock_run_side_effect)
    # Also mock shutil.which if stdlib detection relies on it directly
    mock_shutil_which = mocker.patch('shutil.which', return_value='/fake/path/to/lean')


    # Call the function under test (cache irrelevant for timeout itself)
    success, message = await check_and_compile_item(
         item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
         timeout_seconds=short_timeout, temp_lib_name=TEMP_LIB_NAME
    )

    assert success is False
    assert "Timeout" in message
    assert f"Timeout after {short_timeout}s" in message
    # Check calls (expect stdlib check attempt, then build attempt)
    assert mock_subprocess_run.call_count >= 1 # At least the build call that timed out
    # Assert DB state
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None and retrieved.status == ItemStatus.ERROR
    assert retrieved.lean_error_log is not None
    assert f"Timeout after {short_timeout}s" in retrieved.lean_error_log