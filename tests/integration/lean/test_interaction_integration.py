# File: tests/integration/lean/test_interaction_integration.py

import os
import pathlib
import shutil
import subprocess
import sys
import time
import warnings
from typing import Optional

import pytest

# --- Add project root to path to allow imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir))) # Adjust if structure differs
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Project-Specific Imports ---
try:
    from lean_automator.kb.storage import (
        DEFAULT_DB_PATH, # Import if used directly, otherwise can remove
        ItemStatus,
        ItemType,
        KBItem,
        get_kb_item_by_name,
        initialize_database,
        save_kb_item,
    )
    # Import the public API function
    from lean_automator.lean.interaction import check_and_compile_item
    # REMOVED: Import of internal helper _generate_imports_for_target is no longer needed
    # from lean_automator.lean.interaction import _generate_imports_for_target # <- REMOVED
except ImportError as e:
    print(f"Error importing modules from src/: {e}", file=sys.stderr)
    print(f"Project root added to path: {project_root}", file=sys.stderr)
    raise

# --- Add Config Loader Imports ---
try:
    from lean_automator.config.loader import (
        APP_CONFIG,
        get_lean_automator_shared_lib_path,
    )
except ImportError:
    warnings.warn(
        "lean_automator.config.loader not found. Tests might use fallback config.",
        ImportWarning,
    )
    APP_CONFIG = {}  # Provide fallback empty config
    def get_lean_automator_shared_lib_path() -> Optional[str]:
        return os.getenv("LEAN_AUTOMATOR_SHARED_LIB_PATH")


# --- Configuration ---
LAKE_EXEC_PATH = os.environ.get("LAKE_EXECUTABLE_PATH", "lake") # Match env var name used elsewhere
# Determine Shared Lib Path and Source Dir Name ONCE
# This is crucial as tests rely on this path existing and being configured.
SHARED_LIB_PATH_STR = get_lean_automator_shared_lib_path()
SHARED_LIB_PATH = pathlib.Path(SHARED_LIB_PATH_STR).resolve() if SHARED_LIB_PATH_STR else None

lean_paths_config = APP_CONFIG.get("lean_paths", {})
SHARED_LIB_SRC_DIR: str = lean_paths_config.get("shared_lib_src_dir_name", "VantageLib")

CACHE_ENV_VAR = "LEAN_AUTOMATOR_LAKE_CACHE"


def is_lake_available(lake_exec=LAKE_EXEC_PATH):
    """Check if the lake executable exists."""
    return shutil.which(lake_exec) is not None

def check_shared_lib_path():
    """Check if the shared library path is configured and exists."""
    if not SHARED_LIB_PATH_STR:
        return False, "LEAN_AUTOMATOR_SHARED_LIB_PATH environment variable not set."
    if not SHARED_LIB_PATH or not SHARED_LIB_PATH.is_dir():
        return False, f"Shared library path '{SHARED_LIB_PATH_STR}' does not exist or is not a directory."
    # Check for lakefile existence (optional but good practice)
    if not (SHARED_LIB_PATH / "lakefile.lean").is_file() and \
       not (SHARED_LIB_PATH / "lakefile.toml").is_file():
         warnings.warn(f"Shared library path {SHARED_LIB_PATH} seems to be missing a lakefile.")
    return True, ""


# --- Apply marks ---
shared_lib_ok, shared_lib_reason = check_shared_lib_path()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not is_lake_available(), reason=f"Lake executable '{LAKE_EXEC_PATH}' not found in PATH."),
    pytest.mark.skipif(not shared_lib_ok, reason=f"Shared library path issue: {shared_lib_reason}"),
]

# --- Helper Function ---
def get_expected_path(item_name: str) -> pathlib.Path:
    """Calculates the expected path for an item within the shared library source."""
    assert SHARED_LIB_PATH is not None # Should be checked by pytestmark
    # Derive relative path from unique name, removing shared lib source dir prefix if present
    relative_module_name = item_name
    prefix = SHARED_LIB_SRC_DIR + "."
    if item_name.startswith(prefix):
        relative_module_name = item_name[len(prefix):]

    # Use similar logic to interaction._module_name_to_path
    parts = relative_module_name.split('.')
    if not parts:
        raise ValueError(f"Cannot derive path from empty name parts: {item_name}")
    rel_path = pathlib.Path(*parts).with_suffix(".lean")
    return SHARED_LIB_PATH / SHARED_LIB_SRC_DIR / rel_path


# --- Fixtures ---

@pytest.fixture
def test_db(tmp_path):
    """Fixture for a temporary database per test."""
    db_file = tmp_path / "test_integration_lean.sqlite"
    db_path = str(db_file)
    initialize_database(db_path=db_path)
    yield db_path
    # tmp_path handles cleanup

@pytest.fixture(autouse=True)
def ensure_shared_lib_exists():
    """Fixture to automatically skip if shared lib path isn't valid."""
    if not shared_lib_ok:
        pytest.skip(f"Shared library path issue: {shared_lib_reason}")

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture for a temporary directory to use as LAKE_HOME per test."""
    cache_dir = tmp_path / "lake_cache"
    cache_dir.mkdir()
    yield str(cache_dir)
    # tmp_path handles cleanup

@pytest.fixture(autouse=True)
def cleanup_shared_lib(request):
    """Cleans up specific test files from the shared library after each test."""
    created_files = set()

    # Function to register a file path that might be created
    def add_created_file_path(path: pathlib.Path):
        nonlocal created_files
        if path: # Ensure path is not None
            created_files.add(path.resolve()) # Store resolved path

    # Yield the function to the test
    yield add_created_file_path

    # Teardown: Remove files created during the test
    print(f"\nCleanup: Checking files {created_files}")
    cleaned_parents = set()
    for file_path in created_files:
        if file_path.is_file(): # Check if it's a file before removing
            try:
                parent_dir = file_path.parent
                os.remove(file_path)
                print(f"Cleanup: Removed test file {file_path}")
                cleaned_parents.add(parent_dir)
            except OSError as e:
                print(f"Cleanup Warning: Failed to remove {file_path}: {e}")

    # Optional: Attempt to remove empty parent directories created by tests
    # Be cautious with this, ensure it doesn't remove essential shared lib structure
    shared_lib_src_path = (SHARED_LIB_PATH / SHARED_LIB_SRC_DIR).resolve()
    for parent_dir in sorted(cleaned_parents, key=lambda p: len(p.parts), reverse=True):
        # Only remove if it's inside the source directory and is empty
        if parent_dir.is_relative_to(shared_lib_src_path) and parent_dir != shared_lib_src_path:
             try:
                 if not any(parent_dir.iterdir()): # Check if directory is empty
                     os.rmdir(parent_dir)
                     print(f"Cleanup: Removed empty directory {parent_dir}")
             except OSError as e:
                 print(f"Cleanup Warning: Failed to remove empty dir {parent_dir}: {e}")


# --- Test Cases (Updated for Direct Build Approach) ---

@pytest.mark.asyncio
async def test_compile_success_no_deps_no_cache(test_db, monkeypatch, cleanup_shared_lib):
    """Test compiling simple item directly into shared lib (no cache env var)."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    item_name = f"{SHARED_LIB_SRC_DIR}.Test.NoDeps.NoCache" # Ensure name includes lib prefix
    expected_file_path = get_expected_path(item_name)
    cleanup_shared_lib(expected_file_path) # Register for cleanup

    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def simpleDefNC : Nat := 5", topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(item_name, db_path=test_db)

    assert success is True, f"Compile failed: {message}"
    assert expected_file_path.exists(), "Lean file was not created/kept in shared lib on success."
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved and retrieved.status == ItemStatus.PROVEN

@pytest.mark.asyncio
async def test_compile_success_no_deps_with_cache(test_db, temp_cache_dir, monkeypatch, cleanup_shared_lib):
    """Test compiling simple item directly into shared lib (with cache env var)."""
    monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    item_name = f"{SHARED_LIB_SRC_DIR}.Test.NoDeps.WithCache" # Ensure name includes lib prefix
    expected_file_path = get_expected_path(item_name)
    cleanup_shared_lib(expected_file_path)

    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def simpleDefWC : Nat := 6", topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(item_name, db_path=test_db)

    assert success is True, f"Compile failed: {message}"
    assert expected_file_path.exists(), "Lean file was not created/kept in shared lib on success."
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved and retrieved.status == ItemStatus.PROVEN

@pytest.mark.asyncio
async def test_compile_success_with_deps_cached(test_db, temp_cache_dir, monkeypatch, cleanup_shared_lib):
    """Test compiling item where dependency should be cached by Lake."""
    monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)

    dep_name = f"{SHARED_LIB_SRC_DIR}.Test.DepA.Cached" # Ensure name includes lib prefix
    dep_path = get_expected_path(dep_name)
    cleanup_shared_lib(dep_path)
    dep_item = KBItem(
        unique_name=dep_name, item_type=ItemType.DEFINITION,
        lean_code="def depValueCached : Nat := 10", topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(dep_item, client=None, db_path=test_db)
    # Build the dependency first using the public API
    dep_success, dep_message = await check_and_compile_item(dep_name, db_path=test_db)
    assert dep_success is True, f"Dependency compile failed: {dep_message}"
    assert dep_path.exists(), "Dependency Lean file was not created/kept."

    target_name = f"{SHARED_LIB_SRC_DIR}.Test.TargetB.UsesCached" # Ensure name includes lib prefix
    target_path = get_expected_path(target_name)
    cleanup_shared_lib(target_path)
    # Import name must match the dependency's unique_name
    target_code = (f"import {dep_name}\n\ntheorem useDepCached : depValueCached = 10 := rfl")
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM, lean_code=target_code,
        topic="Test", latex_statement="Dummy", latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    start_time = time.monotonic()
    # Now build the target, Lake should use the already built/cached dependency
    target_success, message = await check_and_compile_item(target_name, db_path=test_db)
    end_time = time.monotonic()
    print(f"\nTarget build time (cached dep): {end_time - start_time:.4f}s")

    assert target_success is True, f"Compilation failed: {message}"
    assert target_path.exists(), "Target Lean file was not created/kept in shared lib on success."
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target and retrieved_target.status == ItemStatus.PROVEN
    # Check dependency status was updated by its compile step
    retrieved_dep = get_kb_item_by_name(dep_name, db_path=test_db)
    assert retrieved_dep and retrieved_dep.status == ItemStatus.PROVEN


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True])
async def test_compile_fail_syntax_error(test_db, temp_cache_dir, monkeypatch, use_cache, cleanup_shared_lib):
    """Test syntax error failure removes the file."""
    if use_cache: monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else: monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"{SHARED_LIB_SRC_DIR}.Test.SyntaxErr.{'Cache' if use_cache else 'NoCache'}" # Ensure name includes lib prefix
    expected_file_path = get_expected_path(item_name)
    # No need to register for cleanup here, as failure *should* remove it.
    # The cleanup fixture will still try, but it should find nothing.

    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION, lean_code="def oops : Nat :=",
        topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(item_name, db_path=test_db)

    assert success is False
    assert not expected_file_path.exists(), "Lean file with syntax error was not removed after failure."
    assert "Lean validation failed" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved and retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert retrieved.lean_error_log and ("error:" in retrieved.lean_error_log or "unknown identifier" in retrieved.lean_error_log) # Check for common error patterns

@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True])
async def test_compile_fail_proof_error(test_db, temp_cache_dir, monkeypatch, use_cache, cleanup_shared_lib):
    """Test proof error failure removes the file."""
    if use_cache: monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else: monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"{SHARED_LIB_SRC_DIR}.Test.ProofErr.{'Cache' if use_cache else 'NoCache'}" # Ensure name includes lib prefix
    expected_file_path = get_expected_path(item_name)
    # No need to register for cleanup

    item = KBItem(
        unique_name=item_name, item_type=ItemType.THEOREM, lean_code="theorem badProof : 1 + 1 = 3 := rfl",
        topic="Test", latex_statement="Dummy", latex_proof="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(item_name, db_path=test_db)

    assert success is False
    assert not expected_file_path.exists(), "Lean file with proof error was not removed after failure."
    assert "Lean validation failed" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved and retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert retrieved.lean_error_log and ("error:" in retrieved.lean_error_log.lower() or "failed to synthesize" in retrieved.lean_error_log) # Check for common proof errors


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [False, True])
async def test_compile_fail_missing_dependency_import(test_db, temp_cache_dir, monkeypatch, use_cache, cleanup_shared_lib):
    """Test failure when an import cannot be resolved by Lake."""
    if use_cache: monkeypatch.setenv(CACHE_ENV_VAR, temp_cache_dir)
    else: monkeypatch.delenv(CACHE_ENV_VAR, raising=False)

    item_name = f"{SHARED_LIB_SRC_DIR}.Test.MissingImport.{'Cache' if use_cache else 'NoCache'}" # Ensure name includes lib prefix
    non_existent_dep_module = f"{SHARED_LIB_SRC_DIR}.Test.DoesNotExistHopefully" # Make name more unique
    expected_file_path = get_expected_path(item_name)
    # No need to register for cleanup

    # Generate the import statement within the code itself
    lean_code = (f"import {non_existent_dep_module}\n\ntheorem usesMissing : True := trivial")
    item = KBItem(
        unique_name=item_name, item_type=ItemType.THEOREM, lean_code=lean_code,
        topic="Test", latex_statement="Dummy", latex_proof="Dummy",
        # Declaring in plan_dependencies is good practice, but the import in lean_code causes the failure
        plan_dependencies=[non_existent_dep_module]
    )
    await save_kb_item(item, client=None, db_path=test_db)

    success, message = await check_and_compile_item(item_name, db_path=test_db)

    assert success is False
    assert not expected_file_path.exists(), "Lean file with missing import was not removed after failure."
    assert "Lean validation failed" in message
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved and retrieved.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert retrieved.lean_error_log is not None
    # Check for various ways Lake/Lean might report this error
    assert ("unknown package" in retrieved.lean_error_log or
            "unknown module" in retrieved.lean_error_log or
            "could not resolve import" in retrieved.lean_error_log or
            "failed to build" in retrieved.lean_error_log or # Lake might report build failure
            non_existent_dep_module in retrieved.lean_error_log), \
           f"Expected import error message not found in log:\n{retrieved.lean_error_log}"

@pytest.mark.asyncio
async def test_compile_success_dep_built_by_lake(test_db, monkeypatch, cleanup_shared_lib):
    """Test Lake successfully building dependency from source present in shared lib."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False) # No cache focus here

    dep_name = f"{SHARED_LIB_SRC_DIR}.Test.DepNeedsBuilding.FromSource" # Ensure name includes lib prefix
    dep_path = get_expected_path(dep_name)
    cleanup_shared_lib(dep_path) # Register dep for potential cleanup

    dep_item = KBItem(
        unique_name=dep_name, item_type=ItemType.DEFINITION,
        lean_code="def depValueNeedsBuildingFS : Nat := 30", topic="Test",
        latex_statement="Dummy", status=ItemStatus.PENDING, # Start as PENDING
    )
    await save_kb_item(dep_item, client=None, db_path=test_db)

    # --- Step 1: Compile the dependency first using the public API ---
    # This will write the dependency's source file to the shared library.
    dep_success, dep_message = await check_and_compile_item(dep_name, db_path=test_db)
    assert dep_success is True, f"Dependency compilation failed unexpectedly: {dep_message}"
    assert dep_path.exists(), "Dependency .lean file should have been created by check_and_compile_item."
    # Verify DB status for dependency is now PROVEN
    retrieved_dep_after_compile = get_kb_item_by_name(dep_name, db_path=test_db)
    assert retrieved_dep_after_compile and retrieved_dep_after_compile.status == ItemStatus.PROVEN
    print(f"\nDependency {dep_name} compiled successfully and file exists at {dep_path}")


    # --- Step 2: Prepare and compile the target item ---
    target_name = f"{SHARED_LIB_SRC_DIR}.Test.TargetUsesBuiltDep.FromSource" # Ensure name includes lib prefix
    target_path = get_expected_path(target_name)
    cleanup_shared_lib(target_path) # Register target for cleanup
    # Generate the import statement for the target's code
    target_code = (f"import {dep_name}\n\ntheorem useBuiltDepFS : depValueNeedsBuildingFS = 30 := rfl")
    target_item = KBItem(
        unique_name=target_name, item_type=ItemType.THEOREM, lean_code=target_code,
        topic="Test", latex_statement="Dummy", latex_proof="Dummy", plan_dependencies=[dep_name]
    )
    await save_kb_item(target_item, client=None, db_path=test_db)

    # --- Step 3: Compile the target; Lake should find and use the existing dependency source/build ---
    start_time = time.monotonic()
    target_success, target_message = await check_and_compile_item(target_name, db_path=test_db)
    end_time = time.monotonic()
    print(f"Target build time (dep built from source): {end_time - start_time:.4f}s")

    assert target_success is True, f"Target compilation failed unexpectedly: {target_message}"
    assert target_path.exists(), "Target lean file should exist on success."
    # Check dependency file still exists (it should)
    assert dep_path.exists(), "Dependency file was unexpectedly removed during target compilation."

    # Verify DB statuses
    retrieved_target = get_kb_item_by_name(target_name, db_path=test_db)
    assert retrieved_target and retrieved_target.status == ItemStatus.PROVEN
    retrieved_dep_final = get_kb_item_by_name(dep_name, db_path=test_db)
    assert retrieved_dep_final and retrieved_dep_final.status == ItemStatus.PROVEN


# --- Configuration Error Tests ---

@pytest.mark.asyncio
async def test_compile_fail_lake_not_found(test_db, cleanup_shared_lib):
    """Test failure when Lake executable path is invalid."""
    item_name = f"{SHARED_LIB_SRC_DIR}.Test.LakeNotFound.Config" # Ensure name includes lib prefix
    # We don't expect the file to be created if lake isn't found early
    # expected_file_path = get_expected_path(item_name)
    # cleanup_shared_lib(expected_file_path) # Not strictly needed

    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION, lean_code="def x := 1",
        topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    invalid_lake_path = "/path/to/nonexistent/lake_executable_qwerty_int_test"
    success, message = await check_and_compile_item(
        item_name, db_path=test_db, lake_executable_path=invalid_lake_path
    )

    assert success is False
    # File should NOT have been written if lake path check happens early,
    # or SHOULD have been removed if written then build fails due to missing lake.
    # Check the calculated path just to be sure
    expected_file_path = get_expected_path(item_name)
    assert not expected_file_path.exists(), "Lean file should not exist after lake not found failure."

    # Error message might come from LeanVerifier init or _run_lake_build
    assert ("Lake executable not found" in message or
            "No such file or directory" in message or
            "Failed to initialize" in message or # Could fail in init
            "Configuration error" in message), \
           f"Unexpected error message: {message}"

    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None
    # Status depends on *when* the failure occurs.
    # If it fails during LeanVerifier init (ValueError), status might remain PENDING.
    # If it fails during _run_lake_build (FileNotFoundError/RuntimeError), status should be ERROR.
    assert retrieved.status in [ItemStatus.ERROR, ItemStatus.PENDING]
    if retrieved.status == ItemStatus.ERROR:
        assert retrieved.lean_error_log is not None
        assert ("Lake executable not found" in retrieved.lean_error_log or
                "No such file or directory" in retrieved.lean_error_log)


@pytest.mark.asyncio
async def test_compile_fail_shared_lib_path_invalid(test_db, monkeypatch):
    """Test failure when LEAN_AUTOMATOR_SHARED_LIB_PATH is invalid during Verifier init."""
    item_name = f"{SHARED_LIB_SRC_DIR}.Test.SharedLibInvalid.Config"
    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION, lean_code="def y := 2",
        topic="Test", latex_statement="Dummy", status=ItemStatus.PENDING # Start as PENDING
    )
    await save_kb_item(item, client=None, db_path=test_db)

    invalid_shared_lib_path = "/path/to/non/existent/shared/lib/qwerty_test"

    # Mock the function used by LeanVerifier.__init__ to get the path
    # Patch within the interaction module where it's imported and called
    monkeypatch.setattr(
        "lean_automator.lean.interaction.get_lean_automator_shared_lib_path",
        lambda: invalid_shared_lib_path,
        raising=True # Ensure the patch takes effect where needed
    )

    # Calling check_and_compile_item should now fail during LeanVerifier init (ValueError)
    success, message = await check_and_compile_item( item_name, db_path=test_db )

    assert success is False
    assert "Configuration error" in message # Check specific error from check_and_compile_item wrapper
    assert ("Shared library path" in message or invalid_shared_lib_path in message) # Check error originates from path validation
    assert "not configured correctly or the directory does not exist" in message

    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None
    # Status should remain PENDING as verification couldn't even start due to config error
    assert retrieved.status == ItemStatus.PENDING
    assert retrieved.lean_error_log is None # No lean error occurred


# --- Timeout Test ---

@pytest.mark.asyncio
async def test_compile_fail_timeout_mocked(test_db, mocker, cleanup_shared_lib):
    """Test failure handling when the direct lake build times out (mocked)."""
    item_name = f"{SHARED_LIB_SRC_DIR}.Test.TimeoutMocked.Direct" # Ensure name includes lib prefix
    expected_file_path = get_expected_path(item_name)
    cleanup_shared_lib(expected_file_path) # Register for cleanup

    item = KBItem(
        unique_name=item_name, item_type=ItemType.DEFINITION,
        lean_code="def anyContent := 1", topic="Test", latex_statement="Dummy"
    )
    await save_kb_item(item, client=None, db_path=test_db)

    short_timeout = 0.1 # Use a very short timeout for testing
    # Module name for the direct build command needs correct calculation
    relative_module_name = item_name
    prefix = SHARED_LIB_SRC_DIR + "."
    if item_name.startswith(prefix):
         relative_module_name = item_name[len(prefix):]
    target_module_name = f"{SHARED_LIB_SRC_DIR}.{relative_module_name}"


    mock_timeout_exception = subprocess.TimeoutExpired(cmd="mock lake build", timeout=short_timeout)
    mock_libdir_result = subprocess.CompletedProcess(args=["lean", "--print-libdir"], returncode=0, stdout="/mock/lean/lib", stderr="")

    # Mock subprocess.run used within the lean_interaction module
    mock_subprocess_run = mocker.patch("lean_automator.lean.interaction.subprocess.run")

    def mock_run_side_effect(*args, **kwargs):
        command_args = kwargs.get("args", args[0] if args else [])
        command_cwd = kwargs.get("cwd")
        # Ensure SHARED_LIB_PATH is resolved for comparison
        resolved_shared_lib_path = str(SHARED_LIB_PATH.resolve()) if SHARED_LIB_PATH else None

        # print(f"Mock subprocess.run: Cmd={command_args}, Cwd={command_cwd}, Expected CWD={resolved_shared_lib_path}") # Debug

        # Handle lean --print-libdir call (part of env setup)
        if "--print-libdir" in command_args:
            # print("Mock: Handling --print-libdir") # Debug
            return mock_libdir_result

        # Handle the specific direct lake build call in the shared lib directory
        # Compare command arguments and CWD carefully
        is_build_command = "build" in command_args
        is_correct_target = command_args[-1] == target_module_name if command_args else False
        is_correct_cwd = str(command_cwd) == resolved_shared_lib_path if command_cwd and resolved_shared_lib_path else False

        if is_build_command and is_correct_target and is_correct_cwd:
             print(f"\nMock: Raising TimeoutExpired for build of '{target_module_name}' in CWD '{command_cwd}'") # Debug
             raise mock_timeout_exception

        # Default for other calls (e.g., 'which', other potential lake calls)
        # print(f"Mock: Handling other command: {command_args}") # Debug
        # Use a real CompletedProcess for default behaviour
        return subprocess.CompletedProcess(args=command_args, returncode=0, stdout=b"Mock output", stderr=b"")


    mock_subprocess_run.side_effect = mock_run_side_effect

    # Mock shutil.which as well, as it might be called during env setup
    mock_which = mocker.patch("lean_automator.lean.interaction.shutil.which")
    def which_side_effect(cmd):
        if cmd == "lean": return "/fake/path/to/lean"
        if cmd == LAKE_EXEC_PATH: return LAKE_EXEC_PATH # Assume lake exists for this test
        return None
    mock_which.side_effect = which_side_effect

    # Call the function under test
    print(f"\nCalling check_and_compile_item for {item_name} with timeout {short_timeout}s")
    success, message = await check_and_compile_item(
        item_name,
        db_path=test_db,
        timeout_seconds=short_timeout, # Pass the short timeout
    )

    assert success is False, f"Expected compilation to fail due to timeout, but got success. Message: {message}"
    assert not expected_file_path.exists(), f"Lean file '{expected_file_path}' should have been removed after timeout failure."

    # Check the failure message returned by check_and_compile_item
    assert ("Timeout" in message or "timed out" in message or "Build execution error" in message), \
        f"Expected timeout message not found in result: {message}"
    # Check specific timeout message if present in LeanVerifier logic
    assert f"timed out after {short_timeout} seconds" in message or "TimeoutExpired" in message

    # Verify the specific build call that should have timed out was attempted
    timeout_call_attempted = False
    resolved_shared_lib_path_str = str(SHARED_LIB_PATH.resolve()) if SHARED_LIB_PATH else None
    for call in mock_subprocess_run.call_args_list:
        args_list = call.kwargs.get("args", call.args[0] if call.args else [])
        call_cwd = call.kwargs.get("cwd")
        if isinstance(args_list, tuple): args_list = list(args_list)

        is_build_command = "build" in args_list
        is_correct_target = args_list[-1] == target_module_name if args_list else False
        is_correct_cwd = str(call_cwd) == resolved_shared_lib_path_str if call_cwd and resolved_shared_lib_path_str else False

        if is_build_command and is_correct_target and is_correct_cwd:
            timeout_call_attempted = True
            print(f"Verified mock call that timed out: args={args_list}, cwd={call_cwd}") # Debug
            break
    assert timeout_call_attempted, f"Expected timeout during the direct build call ('{target_module_name}') in shared lib ('{resolved_shared_lib_path_str}'). Calls made: {mock_subprocess_run.call_args_list}"

    # Assert DB state
    retrieved = get_kb_item_by_name(item_name, db_path=test_db)
    assert retrieved is not None, "Failed to retrieve item from DB after test."
    assert retrieved.status == ItemStatus.ERROR, f"Expected status ERROR after timeout, but got {retrieved.status}."
    assert retrieved.lean_error_log is not None, "Expected lean_error_log to be set after timeout."
    assert ("Timeout" in retrieved.lean_error_log or "timed out" in retrieved.lean_error_log), \
        f"Expected timeout message in lean_error_log. Got: {retrieved.lean_error_log}"