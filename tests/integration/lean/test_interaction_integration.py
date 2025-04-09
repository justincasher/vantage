# File: tests/integration/lean/test_interaction_integration.py

import pytest
import os
import sys
import pathlib
import shutil
import time
import asyncio
import subprocess
import warnings
from unittest.mock import AsyncMock, MagicMock, call # Keep mocks if needed for specific tests

# --- Project-Specific Imports ---
# Adjust path to import from src
# (Assuming project structure allows this)
try:
    from lean_automator.kb.storage import (
        initialize_database,
        save_kb_item,
        get_kb_item_by_name,
        KBItem,
        ItemStatus,
        ItemType
    )
    from lean_automator.lean.interaction import check_and_compile_item 
    # Temporarily import internal helper for one test setup
    from lean_automator.lean.interaction import _generate_imports_for_target
except ImportError as e:
    print(f"Error importing modules: {e}", file=sys.stderr)
    raise

# --- Add Config Loader Imports ---
try:
    from lean_automator.config.loader import APP_CONFIG, get_lean_automator_shared_lib_path
except ImportError:
    warnings.warn("lean_automator.config.loader not found. Tests might use fallback config.", ImportWarning) 
    APP_CONFIG = {} # Provide fallback empty config
    def get_lean_automator_shared_lib_path() -> Optional[str]:
        # Fallback directly to environment variable if config loader is missing
        return os.getenv('LEAN_AUTOMATOR_SHARED_LIB_PATH')

# --- Configuration ---
LAKE_EXEC_PATH = os.environ.get('LAKE_EXECUTABLE', 'lake')
# This should match the hardcoded value in lean_interaction.py
lean_paths_config = APP_CONFIG.get('lean_paths', {})
SHARED_LIB_SRC_DIR: str = lean_paths_config.get('shared_lib_src_dir_name', 'VantageLib') # Default if not in config
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
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
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
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    assert success is True, f"Compile failed: {message}"
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None and retrieved.status == ItemStatus.PROVEN
    # Optionally check for cache files existence if needed, adjusting paths for hardcoded temp lib name


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
        dep_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )
    assert dep_success is True
    # Optionally verify dep olean exists in cache

    # 2. Compile target (Item B depends on A) - should reuse cached dep
    target_name = "Test.TargetB.UsesCached"
    # Import uses the persistent library source directory name (matches lean_interaction.py hardcoding)
    target_code = f"import {SHARED_LIB_SRC_DIR}.Test.DepA.Cached\n\ntheorem useDepCached : depValueCached = 10 := rfl"
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM,
        lean_code=target_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    # Time the second build (optional, can be flaky)
    start_time = time.monotonic()
    target_success, message = await check_and_compile_item(
        target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )
    end_time = time.monotonic()
    print(f"\nTarget build time (cached dep): {end_time - start_time:.4f}s") # Print time for info

    # 3. Assertions for Target B
    assert target_success is True, f"Compilation failed: {message}"
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN
    # Optionally verify target olean exists in cache


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
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
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
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
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
    """Test failure when a declared dependency cannot be resolved by Lake."""
    if use_cache:
        monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else:
        monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"Test.MissingDepLake.{'Cache' if use_cache else 'NoCache'}"
    non_existent_dep = "Test.DoesNotExist"
    # Import uses the persistent library source directory name (matches lean_interaction.py hardcoding)
    lean_code = f"import {SHARED_LIB_SRC_DIR}.Test.DoesNotExist\n\ntheorem usesMissing : True := trivial"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.THEOREM,
        lean_code=lean_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[non_existent_dep] # Declared but won't be found
    )
    await save_kb_item(item, client=None, db_path=test_db)

    # check_and_compile_item will now attempt the build, which should fail in Lake
    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    assert success is False
    # The error should now come from the Lake build process
    assert "Lean validation failed" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None
    # Status should be LEAN_VALIDATION_FAILED because the build attempt failed
    assert retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert retrieved.lean_error_log is not None
    # Check for expected Lake error fragments using the updated assertion
    assert "error:" in retrieved.lean_error_log
    assert "bad import" in retrieved.lean_error_log or non_existent_dep in retrieved.lean_error_log
    assert "no such file or directory" in retrieved.lean_error_log


# --- Tests that rely on specific build behavior ---

@pytest.mark.asyncio
async def test_compile_success_dep_proven_code_exists(test_db, monkeypatch):
    """Test success when dep status=PROVEN and its code exists in shared lib."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    dep_name = "Test.DepProvenCodeExists.NoCache"
    dep_item = KBItem(
        unique_name=dep_name, item_type=ItemType.DEFINITION,
        lean_code="def depValueCodeExistsNC : Nat := 20", topic="Test",
        latex_statement="Dummy", status=ItemStatus.PROVEN # Status is PROVEN
    )
    # Compile the dependency first to ensure its code is in the shared library
    # This implicitly uses the hardcoded names inside check_and_compile_item
    await save_kb_item(dep_item, client=None, db_path=test_db)
    dep_success, dep_msg = await check_and_compile_item(
        dep_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )
    assert dep_success is True, f"Dependency compilation failed: {dep_msg}"

    target_name = "Test.TargetDepProvenCodeExists.NoCache"
    # Import uses the persistent library source directory name (matches lean_interaction.py hardcoding)
    target_code = f"import {SHARED_LIB_SRC_DIR}.Test.DepProvenCodeExists.NoCache\n\ntheorem useVerifiedCodeExistsNC : depValueCodeExistsNC = 20 := rfl"
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM,
        lean_code=target_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(
        target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    assert success is True, f"Compilation failed unexpectedly: {message}"
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN
    retrieved_dep = get_kb_item_by_name(dep_name, db_path=test_db)
    assert retrieved_dep is not None # Verify dep still exists


@pytest.mark.asyncio
async def test_compile_success_dep_built_by_lake(test_db, monkeypatch):
    """Test Lake successfully building dependency from source present in shared lib. Cache unset."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    dep_name = "Test.DepNeedsBuilding.NoCache"
    dep_item = KBItem(
        unique_name=dep_name, item_type=ItemType.DEFINITION,
        lean_code="def depValueNeedsBuildingNC : Nat := 30", topic="Test",
        latex_statement="Dummy", status=ItemStatus.PENDING # Status is NOT PROVEN initially
    )
    # Save the dependency item to the DB (status PENDING)
    await save_kb_item(dep_item, client=None, db_path=test_db)

    # Manually place the dependency's source code in the expected shared library location
    # using the hardcoded source directory name
    shared_lib_path_str = get_lean_automator_shared_lib_path()
    if shared_lib_path_str:
        shared_lib_path = pathlib.Path(shared_lib_path_str).resolve()
        dep_rel_path = pathlib.Path(*dep_name.split('.')).with_suffix('.lean')
        # Construct destination path using the hardcoded source dir name
        dep_dest_path = shared_lib_path / SHARED_LIB_SRC_DIR / dep_rel_path
        dep_dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Generate the full code including potential imports for the dependency
        import_block = _generate_imports_for_target(dep_item)
        separator = "\n\n" if import_block and dep_item.lean_code else ""
        full_dep_code = f"{import_block}{separator}{str(dep_item.lean_code)}" # Ensure lean_code is str
        dep_dest_path.write_text(full_dep_code, encoding='utf-8')
        print(f"\nManually placed dependency code at: {dep_dest_path}") # For debugging
    else:
        pytest.skip("LEAN_AUTOMATOR_SHARED_LIB_PATH not set, cannot place dependency code manually.")


    target_name = "Test.TargetUsesBuiltDep.NoCache"
    # Import uses the persistent library source directory name (matches lean_interaction.py hardcoding)
    target_code = f"import {SHARED_LIB_SRC_DIR}.Test.DepNeedsBuilding.NoCache\n\ntheorem useBuiltDepNC : depValueNeedsBuildingNC = 30 := rfl"
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM,
        lean_code=target_code, topic="Test", latex_statement="Dummy",
        latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    # Now compile the target; Lake should find and build the dependency from the source we placed
    success, message = await check_and_compile_item(
        target_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH
    )

    assert success is True, f"Compilation failed unexpectedly: {message}"
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target is not None and retrieved_target.status == ItemStatus.PROVEN
    # Verify dep status remains unchanged in DB, as it wasn't processed directly
    retrieved_dep = get_kb_item_by_name(dep_name, db_path=test_db)
    assert retrieved_dep is not None and retrieved_dep.status == ItemStatus.PENDING


# --- Configuration Error Tests ---

@pytest.mark.asyncio
async def test_compile_fail_lake_not_found(test_db):
    """Test failure when Lake executable path is invalid."""
    item_name = "Test.LakeNotFound.Config"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def x := 1", topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    invalid_lake_path = "/path/to/nonexistent/lake_executable_qwerty_int"
    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=invalid_lake_path
    )

    assert success is False
    # The function should detect this early if possible, or fail during subprocess call
    assert "Lake executable not found" in message or "No such file or directory" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None
    # Status might remain PENDING or become ERROR depending on when the check happens
    assert retrieved.status in [ItemStatus.PENDING, ItemStatus.ERROR]


# --- Timeout Test (Mocked in integration as it's hard to reliably trigger) ---

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
    mock_timeout_exception = subprocess.TimeoutExpired(cmd="mock lake build", timeout=short_timeout)
    mock_libdir_result = subprocess.CompletedProcess(args=['lean', '--print-libdir'], returncode=0, stdout='/mock/lean/lib', stderr='')

    def mock_run_side_effect(*args, **kwargs):
        command_args = kwargs.get('args', args[0] if args else [])
        # print(f"Mock subprocess.run called with: {command_args}") # Uncomment for debug
        # Handle lean --print-libdir call
        if '--print-libdir' in command_args:
            return mock_libdir_result
        # Handle the specific temporary lake build call (simulate timeout)
        if 'build' in command_args and command_args[-1].startswith('TempVerifyLib.'):
             raise mock_timeout_exception
        # Default for other calls (e.g., persistent build, 'which')
        return subprocess.CompletedProcess(args=command_args, returncode=0, stdout='Mock output', stderr='')

    # Mock subprocess.run used within the lean_interaction module
    mock_subprocess_run = mocker.patch('lean_automator.lean.interaction.subprocess.run', side_effect=mock_run_side_effect)
    # Mock shutil.which if stdlib detection relies on it directly
    mock_shutil_which = mocker.patch('lean_automator.lean.interaction.shutil.which', return_value='/fake/path/to/lean')


    # Call the function under test
    success, message = await check_and_compile_item(
         item_name, db_path=test_db, lake_executable_path=LAKE_EXEC_PATH,
         timeout_seconds=short_timeout
    )

    assert success is False
    assert "Timeout" in message
    assert f"Timeout after {short_timeout}s" in message
    # Check subprocess calls were made
    assert mock_subprocess_run.call_count >= 1
    # Specifically check if the build command that times out was attempted
    timeout_call_found = False
    for call_args in mock_subprocess_run.call_args_list:
        args_list = call_args.kwargs.get('args', call_args.args[0] if call_args.args else [])
        if isinstance(args_list, tuple): args_list = list(args_list) # Handle if args passed as tuple
        # Check if it's the build command for the temporary library
        if 'build' in args_list and args_list[-1].startswith('TempVerifyLib.'):
            timeout_call_found = True
            break
    assert timeout_call_found, "Expected timeout during the temporary verification build call"

    # Assert DB state
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None and retrieved.status == ItemStatus.ERROR # Status should be ERROR on timeout
    assert retrieved.lean_error_log is not None
    assert f"Timeout after {short_timeout}s" in retrieved.lean_error_log