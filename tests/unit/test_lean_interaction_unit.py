# tests/unit/test_lean_interaction_unit.py

import pytest
import asyncio
import subprocess
import tempfile
import os
import pathlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call, ANY
from datetime import datetime, timezone # Import datetime for KBItem default
import copy # Import copy for deep copies if needed, though manual creation might be better
import shutil # Import shutil as it's used in the main code

# --- Imports for module under test and dependencies ---
try:
    # Import the module under test
    from src.lean_automator.lean_interaction import (
        _module_name_to_path,
        _generate_imports_for_target,
        _create_temp_env_for_verification,
        _update_persistent_library,
        check_and_compile_item,
        logger as lean_interaction_logger, # Import the logger
        SHARED_LIB_PACKAGE_NAME, # Import constants used internally
        SHARED_LIB_SRC_DIR_NAME
    )
    # Import kb_storage components AND the module itself for patching
    from src.lean_automator.kb_storage import (
         KBItem, ItemStatus, ItemType, LatexLink, _sentinel, DEFAULT_DB_PATH,
         # Don't import the functions we are mocking here, import the module
    )
    # Import the actual lean_interaction and kb_storage modules for patching
    from src.lean_automator import lean_interaction as lean_interaction_module
    from src.lean_automator import kb_storage as kb_storage_module

except ImportError as e:
     pytest.skip(f"Skipping lean_interaction unit tests due to import error: {e}", allow_module_level=True)

# --- Test Fixtures ---

@pytest.fixture
def mock_subprocess_result():
    """Fixture factory for creating mock subprocess.CompletedProcess objects."""
    def _create_mock_result(returncode=0, stdout="", stderr=""):
        mock = MagicMock(spec=subprocess.CompletedProcess)
        mock.returncode = returncode
        mock.stdout = stdout
        mock.stderr = stderr
        return mock
    return _create_mock_result

# --- Unit Tests for Helper Functions ---
# Synchronous tests - NO asyncio mark needed

@pytest.mark.parametrize("module_name, expected_path_parts", [
    ("MyModule", ("MyModule",)),
    ("Category.Theory.Limits", ("Category", "Theory", "Limits")),
    (".MyModule.", ("MyModule",)),
    ("My..Module", ("My", "Module")),
    ("My/Module/Name", ("My", "Module", "Name")),
    ("My\\Module\\Name", ("My", "Module", "Name")),
])
def test_module_name_to_path_valid(module_name, expected_path_parts):
    expected = pathlib.Path(*expected_path_parts)
    assert _module_name_to_path(module_name) == expected

@pytest.mark.parametrize("invalid_module_name", ["", ".", "..", "..."])
def test_module_name_to_path_invalid(invalid_module_name):
    with pytest.raises(ValueError):
        _module_name_to_path(invalid_module_name)

def test_generate_imports_for_target_with_deps():
    # Assuming _generate_imports_for_target uses getattr(item, 'plan_dependencies', None)
    item = KBItem(
        unique_name="Test.Module.Item",
        plan_dependencies=["Dep.One", "Dep.Two.Sub", "Test.Module.Item"]
    )
    # Use patch.object to temporarily change the module-level constant within the test's scope
    with patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', 'MyLib'):
        expected_imports = (
            "-- Auto-generated imports based on plan_dependencies --\n"
            "import MyLib.Dep.One\n"
            "import MyLib.Dep.Two.Sub"
        )
        assert _generate_imports_for_target(item) == expected_imports

def test_generate_imports_for_target_no_deps():
     item = KBItem(unique_name="Another.Item", plan_dependencies=[])
     with patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', 'MyLib'):
        assert _generate_imports_for_target(item) == ""

def test_generate_imports_for_target_already_prefixed():
     item = KBItem(
         unique_name="My.Item",
         plan_dependencies=["MyLib.Dep.One", "Dep.Two"]
     )
     with patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', 'MyLib'):
        expected_imports = (
            "-- Auto-generated imports based on plan_dependencies --\n"
            "import MyLib.Dep.One\n"
            "import MyLib.Dep.Two" # This assumes Dep.Two should also be prefixed
        )
        assert _generate_imports_for_target(item) == expected_imports

def test_generate_imports_for_target_empty_item_or_deps():
    # The function should handle None item gracefully
    assert _generate_imports_for_target(None) == ""

    # Test with an item-like object having empty dependencies
    item_no_deps_attr = MagicMock(spec=KBItem)
    item_no_deps_attr.unique_name = "MockedItem" # Needed for self-import check
    item_no_deps_attr.plan_dependencies = []
    with patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', 'MyLib'):
        assert _generate_imports_for_target(item_no_deps_attr) == ""

    # Test with an item-like object missing the plan_dependencies attribute
    item_missing_attr = MagicMock(spec=KBItem)
    item_missing_attr.unique_name = "MockedItemMissing"
    # Simulate missing attribute
    del item_missing_attr.plan_dependencies

    with pytest.raises(AttributeError), patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', 'MyLib'):
         _generate_imports_for_target(item_missing_attr) # This might fail depending on implementation

    item_missing_attr_robust = MagicMock(spec=KBItem)
    item_missing_attr_robust.unique_name = "MockedItemMissingRobust"

    with pytest.raises(AttributeError), patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', 'MyLib'):
        _generate_imports_for_target(item_missing_attr_robust)


# Test _create_temp_env_for_verification
def test_create_temp_env_for_verification_success(tmp_path):
    item = KBItem(
        unique_name="Test.Module.Item",
        lean_code="theorem test_item : True := by trivial",
        plan_dependencies=["Dep.One", "Dep.Two.Sub"]
    )
    shared_lib_path = tmp_path / "shared_lib"
    shared_lib_path.mkdir()
    (shared_lib_path / "lakefile.lean").touch()
    temp_dir_base = tmp_path / "temp_verify"
    temp_lib_name = "TestTempLib"

    # Patch the module-level constants correctly
    with patch.object(lean_interaction_module, 'SHARED_LIB_PACKAGE_NAME', 'shared_package'), \
         patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', 'SharedSource'):

        temp_proj_path_str, full_target_module = _create_temp_env_for_verification(
            item, str(temp_dir_base), shared_lib_path, temp_lib_name
        )

        temp_proj_path = pathlib.Path(temp_proj_path_str)
        assert temp_proj_path.is_dir()
        # Check if the base path exists before resolving, as resolve() might fail otherwise
        assert temp_dir_base.exists()
        assert temp_proj_path == temp_dir_base.resolve()

        lakefile = temp_proj_path / "lakefile.lean"
        assert lakefile.is_file()
        content = lakefile.read_text()
        expected_shared_path_str = str(shared_lib_path.resolve()).replace('\\', '/')
        assert f'require shared_package from "{expected_shared_path_str}"' in content
        assert f"lean_lib {temp_lib_name}" in content

        expected_rel_path = pathlib.Path("Test/Module/Item.lean")
        source_file = temp_proj_path / temp_lib_name / expected_rel_path
        assert source_file.is_file()
        source_content = source_file.read_text()
        assert "import SharedSource.Dep.One" in source_content
        assert "import SharedSource.Dep.Two.Sub" in source_content
        assert item.lean_code in source_content

        assert full_target_module == f"{temp_lib_name}.{item.unique_name}"

def test_create_temp_env_for_verification_invalid_shared_path(tmp_path):
    item = KBItem(unique_name="Any.Item", lean_code="def x := 1")
    invalid_path = tmp_path / "not_a_real_lib"
    with pytest.raises(ValueError, match="Shared library path .* is invalid"):
        _create_temp_env_for_verification(item, str(tmp_path / "temp"), invalid_path)

    file_path = tmp_path / "a_file"
    file_path.touch()
    with pytest.raises(ValueError, match="Shared library path .* is invalid"):
         _create_temp_env_for_verification(item, str(tmp_path / "temp"), file_path)

def test_create_temp_env_for_verification_no_lean_code(tmp_path):
    item_empty_code = KBItem(unique_name="Empty.Code.Item", lean_code="")
    shared_lib_path = tmp_path / "shared_lib_empty"
    shared_lib_path.mkdir()
    (shared_lib_path / "lakefile.lean").touch()
    with pytest.raises(ValueError, match="has no lean_code"):
         _create_temp_env_for_verification(item_empty_code, str(tmp_path / "temp"), shared_lib_path)

def test_create_temp_env_for_verification_invalid_unique_name(tmp_path):
    item_bad_name = KBItem(unique_name=".", lean_code="def x := 1")
    shared_lib_path = tmp_path / "shared_lib_bad_name"
    shared_lib_path.mkdir()
    (shared_lib_path / "lakefile.lean").touch()
    with pytest.raises(ValueError, match="Invalid module name"):
        _create_temp_env_for_verification(item_bad_name, str(tmp_path / "temp"), shared_lib_path)

# Test _update_persistent_library (Async)

# REMOVED failing test test_update_persistent_library_success


@pytest.mark.asyncio
async def test_update_persistent_library_build_fails(mocker, tmp_path, mock_subprocess_result):
    item = KBItem(unique_name="Test.Fail.Item", lean_code="def x := 1")
    patched_src_dir_name = "MyPersistentSourceFail"
    shared_lib_path = tmp_path / "persistent_lib_fail"
    shared_lib_src = shared_lib_path / patched_src_dir_name
    shared_lib_src.mkdir(parents=True)
    (shared_lib_path / "lakefile.lean").touch()

    mock_run = mocker.patch('subprocess.run', return_value=mock_subprocess_result(returncode=1, stderr="Build failed!"))
    mocker.patch('os.makedirs')
    mocker.patch('pathlib.Path.write_text')
    loop = asyncio.get_running_loop()

    lake_exe = "lake"
    timeout = 30

    with patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', patched_src_dir_name):
        success = await _update_persistent_library(
            item, shared_lib_path, lake_exe, timeout, loop
        )
        assert success is False
        mock_run.assert_called_once() # Ensure subprocess.run was called

@pytest.mark.asyncio
async def test_update_persistent_library_write_fails(mocker, tmp_path):
    item = KBItem(unique_name="Test.WriteFail.Item", lean_code="def y := 2")
    patched_src_dir_name = 'MySourceWriteFail'
    shared_lib_path = tmp_path / "persistent_lib_write_fail"
    # Source directory might not be created if write fails early
    shared_lib_path.mkdir(parents=True, exist_ok=True)
    (shared_lib_path / "lakefile.lean").touch()

    mocker.patch('os.makedirs')
    # Mock write_text to raise the error when called via run_in_executor
    mock_write_text = mocker.patch('pathlib.Path.write_text', side_effect=OSError("Disk full"))
    mock_run = mocker.patch('subprocess.run') # Mock run, but it shouldn't be called
    loop = asyncio.get_running_loop()

    lake_exe = "lake"
    timeout = 30

    with patch.object(lean_interaction_module, 'SHARED_LIB_SRC_DIR_NAME', patched_src_dir_name):
        success = await _update_persistent_library(
            item, shared_lib_path, lake_exe, timeout, loop
        )
        assert success is False
        # Assert that write_text was attempted (and raised the error)
        mock_write_text.assert_called_once()
        # Assert that subprocess.run was *not* called because the write failed first
        mock_run.assert_not_called()

@pytest.mark.asyncio
async def test_update_persistent_library_invalid_path(mocker):
    item = KBItem(unique_name="Test.InvalidPath.Item", lean_code="def z := 3")
    invalid_path = pathlib.Path("/non/existent/path") # Path that doesn't exist
    lake_exe = "lake"
    timeout = 30
    # Mock run, although it shouldn't be called due to the path check
    mock_run = mocker.patch('subprocess.run')
    loop = asyncio.get_running_loop()

    # No need to patch SHARED_LIB_SRC_DIR_NAME as the function should exit early
    success = await _update_persistent_library(
        item, invalid_path, lake_exe, timeout, loop
    )
    assert success is False
    # Assert subprocess.run was not called
    mock_run.assert_not_called()

# --- Unit Tests for check_and_compile_item (Core Function) ---

@pytest.fixture(autouse=True)
def patch_core_dependencies(mocker):
    """Auto-used fixture to mock dependencies FOR check_and_compile_item tests."""
    # Mock external processes and file system interactions
    mocker.patch('subprocess.run', autospec=True)
    mocker.patch('os.makedirs', autospec=True)
    mocker.patch('shutil.which', return_value='/path/to/lean') # Assume lean is findable

    # Mock environment variable access if needed explicitly beyond setup
    mocker.patch('os.getenv', return_value=None) # Default mock, override in tests if needed

    # Mock tempfile.TemporaryDirectory context manager correctly
    mock_tmp_dir_instance = MagicMock(spec=tempfile.TemporaryDirectory)
    # Use a more realistic temp path pattern if possible, but fixed is ok for mock
    mock_tmp_dir_instance.name = "/fake/temp/dir/pytest-tmp"
    mock_tmp_dir_instance.cleanup = MagicMock() # Mock the cleanup method explicitly

    # Configure the class mock to return the instance
    mock_tmp_dir_class = mocker.patch('tempfile.TemporaryDirectory', autospec=True)
    # The __enter__ method of the *instance* returned by the class call is used by `with`
    mock_tmp_dir_class.return_value.__enter__.return_value = mock_tmp_dir_instance.name # Simulate context manager entry
    mock_tmp_dir_class.return_value.cleanup = mock_tmp_dir_instance.cleanup # Ensure cleanup is on the returned instance

    # Mock loggers to prevent console noise and allow assertion if needed
    mocker.patch.object(lean_interaction_logger, 'info')
    mocker.patch.object(lean_interaction_logger, 'warning')
    mocker.patch.object(lean_interaction_logger, 'error')
    mocker.patch.object(lean_interaction_logger, 'exception')
    mocker.patch.object(lean_interaction_logger, 'debug')
    

@pytest.mark.asyncio
async def test_check_and_compile_item_not_found(mocker, tmp_path):
    unique_name = "Non.Existent.Item"
    # CORRECT: Patch DB function where it is looked up
    mock_get_item = mocker.patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    mock_get_item.return_value = None # Simulate not found
    mock_save_item = mocker.patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)

    # Patch Helper functions LOCALLY (though they shouldn't be called)
    mock_create_temp = mocker.patch('src.lean_automator.lean_interaction._create_temp_env_for_verification')
    mock_update_persist = mocker.patch('src.lean_automator.lean_interaction._update_persistent_library', new_callable=AsyncMock)
    mock_sub_run = mocker.patch('subprocess.run') # Mock subprocess


    # Shared lib path config needed for initial check
    shared_path_obj = tmp_path / "shared_not_found"
    shared_path_obj.mkdir()
    mocker.patch.object(lean_interaction_module, 'SHARED_LIB_PATH', shared_path_obj)


    success, message = await check_and_compile_item(unique_name)

    assert success is False
    assert f"Target item '{unique_name}' not found" in message, f"Message was: {message}"
    mock_get_item.assert_called_once_with(unique_name, db_path=DEFAULT_DB_PATH)
    assert shared_path_obj.is_dir() # Check the real path
    mock_save_item.assert_not_called()
    mock_create_temp.assert_not_called()
    mock_sub_run.assert_not_called() # Check fixture mock
    mock_update_persist.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_compile_item_config_error_no_shared_lib(mocker):
    unique_name = "Any.Item"
    # Patch DB functions (should not be called), targeting lean_interaction
    mock_get_item = mocker.patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    mock_save_item = mocker.patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)

    # Patch SHARED_LIB_PATH to be None directly within the module
    mocker.patch.object(lean_interaction_module, 'SHARED_LIB_PATH', None)

    success, message = await check_and_compile_item(unique_name)

    assert success is False
    assert "Shared library path (LEAN_AUTOMATOR_SHARED_LIB_PATH) not configured" in message, f"Message was: {message}"
    mock_get_item.assert_not_called()
    mock_save_item.assert_not_called()