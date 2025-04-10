# tests/unit/lean/test_interaction_unit.py

import asyncio
import pathlib
import subprocess
import tempfile

# Keep patch from unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Imports for module under test and dependencies ---
try:
    # Import the module under test
    # Import for patching APP_CONFIG or related accessors if needed elsewhere
    # from src.lean_automator.config import (
    #     loader as config_loader,
    # )  # Using alias for consistency if needed
    # from src.lean_automator.kb import storage as kb_storage_module # noqa: F401

    # Import kb_storage components AND the module itself for patching
    from src.lean_automator.kb.storage import (
        DEFAULT_DB_PATH,
        # ItemStatus,
        # ItemType,
        KBItem,
        # LatexLink,
        # _sentinel,
        # Don't import the functions we are mocking here, import the module
        # get_kb_item_by_name, # Needed for mocks in check_and_compile tests
        # save_kb_item, # Needed for mocks in check_and_compile tests
    )

    # Import the actual lean_interaction and kb_storage modules for patching
    from src.lean_automator.lean import interaction as lean_interaction_module
    from src.lean_automator.lean.interaction import (
        _create_temp_env_for_verification,
        _generate_imports_for_target,
        _module_name_to_path,
        _update_persistent_library,
        check_and_compile_item,
    )
    from src.lean_automator.lean.interaction import (
        logger as lean_interaction_logger,  # Import the logger
        # Module-level constants will be patched via the module object
    )

except ImportError as e:
    pytest.skip(
        f"Skipping lean_interaction unit tests due to import error: {e}",
        allow_module_level=True,
    )

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


@pytest.mark.parametrize(
    "module_name, expected_path_parts",
    [
        ("MyModule", ("MyModule",)),
        ("Category.Theory.Limits", ("Category", "Theory", "Limits")),
        (".MyModule.", ("MyModule",)),
        ("My..Module", ("My", "Module")),
        ("My/Module/Name", ("My", "Module", "Name")),
        ("My\\Module\\Name", ("My", "Module", "Name")),
    ],
)
def test_module_name_to_path_valid(module_name, expected_path_parts):
    expected = pathlib.Path(*expected_path_parts)
    assert _module_name_to_path(module_name) == expected


@pytest.mark.parametrize("invalid_module_name", ["", ".", "..", "..."])
def test_module_name_to_path_invalid(invalid_module_name):
    with pytest.raises(ValueError):
        _module_name_to_path(invalid_module_name)


# Use standard patch.object from unittest.mock
def test_generate_imports_for_target_with_deps():
    # Assuming _generate_imports_for_target uses
    # getattr(item, 'plan_dependencies', None)
    item = KBItem(
        unique_name="Test.Module.Item",
        plan_dependencies=[
            "Dep.One",
            "Dep.Two.Sub",
            "Test.Module.Item",
        ],  # Self-import should be excluded
    )
    # Patch the module-level constant directly for the test's scope using patch.object
    with patch.object(lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", "MyLib"):
        expected_imports = (
            "-- Auto-generated imports based on plan_dependencies --\n"
            "import MyLib.Dep.One\n"
            "import MyLib.Dep.Two.Sub"  # Function sorts imports alphabetically
        )
        assert _generate_imports_for_target(item) == expected_imports


# Use standard patch.object from unittest.mock
def test_generate_imports_for_target_no_deps():
    item = KBItem(unique_name="Another.Item", plan_dependencies=[])
    # Patch even if no deps, to ensure behavior is consistent regardless
    # of actual config
    with patch.object(lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", "MyLib"):
        assert _generate_imports_for_target(item) == ""


# Use standard patch.object from unittest.mock
def test_generate_imports_for_target_already_prefixed():
    item = KBItem(
        unique_name="My.Item",
        plan_dependencies=["MyLib.Dep.One", "Dep.Two"],  # One prefixed, one not
    )
    # Patch the module-level constant directly using patch.object
    with patch.object(lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", "MyLib"):
        expected_imports = (
            "-- Auto-generated imports based on plan_dependencies --\n"
            "import MyLib.Dep.One\n"  # This one should be included as is
            "import MyLib.Dep.Two"  # This one should get the MyLib prefix
        )
        # The function sorts the imports, which happens to match the order here
        assert _generate_imports_for_target(item) == expected_imports


# Use standard patch.object from unittest.mock
def test_generate_imports_for_target_empty_item_or_deps():
    # The function should handle None item gracefully
    # Need to patch context even for None, as the constant might be accessed
    # before None check
    with patch.object(lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", "MyLib"):
        assert _generate_imports_for_target(None) == ""

    # Test with an item-like object having empty dependencies
    item_no_deps_attr = MagicMock(spec=KBItem)
    item_no_deps_attr.unique_name = "MockedItem"  # Needed for self-import check
    item_no_deps_attr.plan_dependencies = []
    with patch.object(lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", "MyLib"):
        assert _generate_imports_for_target(item_no_deps_attr) == ""

    # Test with an item-like object missing the plan_dependencies attribute
    item_missing_attr = MagicMock(spec=KBItem)
    item_missing_attr.unique_name = "MockedItemMissing"
    # Simulate missing attribute
    with pytest.raises(AttributeError), patch.object(
        lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", "MyLib"
    ):
        # Ensure access triggers error if missing and not handled by
        # getattr(..., []) etc.
        mock_instance_missing_attr = MagicMock(
            spec=["unique_name"]
        )  # Only spec unique_name
        mock_instance_missing_attr.unique_name = "ItemMissingDepsAttr"
        _generate_imports_for_target(mock_instance_missing_attr)


# Test _create_temp_env_for_verification
# Use standard patch.object from unittest.mock
def test_create_temp_env_for_verification_success(
    tmp_path,
):  # No mocker needed here if only using patch.object
    item = KBItem(
        unique_name="Test.Module.Item",
        lean_code="theorem test_item : True := by trivial",
        plan_dependencies=["Dep.One", "Dep.Two.Sub"],
    )
    shared_lib_path = tmp_path / "shared_lib"
    shared_lib_path.mkdir()
    (shared_lib_path / "lakefile.lean").touch()
    temp_dir_base = tmp_path / "temp_verify"
    temp_lib_name = "TestTempLib"

    # Define the test-specific values for patching
    test_package_name = "shared_package"
    test_src_dir_name = "SharedSource"

    # Patch the module-level constants directly for the test's scope using patch.object
    # Stack context managers using commas
    with patch.object(
        lean_interaction_module, "SHARED_LIB_PACKAGE_NAME", test_package_name
    ), patch.object(
        lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", test_src_dir_name
    ):
        temp_proj_path_str, full_target_module = _create_temp_env_for_verification(
            item, str(temp_dir_base), shared_lib_path, temp_lib_name
        )

        temp_proj_path = pathlib.Path(temp_proj_path_str)
        assert temp_proj_path.is_dir()
        # Check if the base path exists before resolving, as resolve() might fail
        # otherwise
        assert temp_dir_base.exists()
        assert temp_proj_path == temp_dir_base.resolve()

        lakefile = temp_proj_path / "lakefile.lean"
        assert lakefile.is_file()
        content = lakefile.read_text()
        expected_shared_path_str = str(shared_lib_path.resolve()).replace("\\", "/")
        # Assert using the patched package name
        assert (
            f'require {test_package_name} from "{expected_shared_path_str}"' in content
        )
        assert f"lean_lib {temp_lib_name}" in content

        expected_rel_path = pathlib.Path("Test/Module/Item.lean")
        source_file = temp_proj_path / temp_lib_name / expected_rel_path
        assert source_file.is_file()
        source_content = source_file.read_text()
        # Assert imports using the patched source directory name
        assert f"import {test_src_dir_name}.Dep.One" in source_content
        assert f"import {test_src_dir_name}.Dep.Two.Sub" in source_content
        assert item.lean_code in source_content

        assert full_target_module == f"{temp_lib_name}.{item.unique_name}"


def test_create_temp_env_for_verification_invalid_shared_path(tmp_path):
    item = KBItem(unique_name="Any.Item", lean_code="def x := 1")
    invalid_path = tmp_path / "not_a_real_lib"
    # No patching needed here as the error should occur before using patched constants
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
    # No patching needed
    with pytest.raises(ValueError, match="has no lean_code"):
        _create_temp_env_for_verification(
            item_empty_code, str(tmp_path / "temp"), shared_lib_path
        )


def test_create_temp_env_for_verification_invalid_unique_name(tmp_path):
    item_bad_name = KBItem(unique_name=".", lean_code="def x := 1")
    shared_lib_path = tmp_path / "shared_lib_bad_name"
    shared_lib_path.mkdir()
    (shared_lib_path / "lakefile.lean").touch()
    # No patching needed
    with pytest.raises(ValueError, match="Invalid module name"):
        _create_temp_env_for_verification(
            item_bad_name, str(tmp_path / "temp"), shared_lib_path
        )


# Test _update_persistent_library (Async)


# Use standard patch.object from unittest.mock
# Keep mocker fixture for other mocks (subprocess.run etc.)
@pytest.mark.asyncio
async def test_update_persistent_library_build_fails(
    mocker, tmp_path, mock_subprocess_result
):
    item = KBItem(unique_name="Test.Fail.Item", lean_code="def x := 1")
    patched_src_dir_name = "MyPersistentSourceFail"
    shared_lib_path = tmp_path / "persistent_lib_fail"
    shared_lib_src_within_path = (
        shared_lib_path / patched_src_dir_name
    )  # Actual target dir for file write
    shared_lib_src_within_path.mkdir(parents=True)
    (shared_lib_path / "lakefile.lean").touch()

    # Use mocker for non-object patches if needed
    mock_run = mocker.patch(
        "subprocess.run",
        return_value=mock_subprocess_result(returncode=1, stderr="Build failed!"),
    )
    mock_makedirs = mocker.patch(
        "os.makedirs"
    )  # Mock os.makedirs used via run_in_executor
    mock_write_text = mocker.patch(
        "pathlib.Path.write_text"
    )  # Mock write_text used via run_in_executor
    loop = asyncio.get_running_loop()

    lake_exe = "lake"
    timeout = 30

    # Patch the constant using patch.object
    with patch.object(
        lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", patched_src_dir_name
    ):
        success = await _update_persistent_library(
            item, shared_lib_path, lake_exe, timeout, loop
        )
        assert success is False
        mock_makedirs.assert_called_once()  # Should be called to ensure dir exists
        mock_write_text.assert_called_once()  # Should be called before build attempt
        mock_run.assert_called_once()  # Ensure subprocess.run was called for the build


# Use standard patch.object from unittest.mock
# Keep mocker fixture for other mocks
@pytest.mark.asyncio
async def test_update_persistent_library_write_fails(mocker, tmp_path):
    item = KBItem(unique_name="Test.WriteFail.Item", lean_code="def y := 2")
    patched_src_dir_name = "MySourceWriteFail"
    shared_lib_path = tmp_path / "persistent_lib_write_fail"
    shared_lib_path.mkdir(parents=True, exist_ok=True)
    (shared_lib_path / "lakefile.lean").touch()

    # Use mocker for non-object patches
    mock_makedirs = mocker.patch("os.makedirs")
    mock_write_lambda_target = mocker.patch(
        "pathlib.Path.write_text", side_effect=OSError("Disk full")
    )
    mock_run = mocker.patch("subprocess.run")  # Mock run, but it shouldn't be called
    loop = asyncio.get_running_loop()

    lake_exe = "lake"
    timeout = 30

    # Patch the constant using patch.object
    with patch.object(
        lean_interaction_module, "SHARED_LIB_SRC_DIR_NAME", patched_src_dir_name
    ):
        success = await _update_persistent_library(
            item, shared_lib_path, lake_exe, timeout, loop
        )
        assert success is False
        # Assert that makedirs was attempted
        mock_makedirs.assert_called_once()
        # Assert that write_text was attempted (and raised the error)
        mock_write_lambda_target.assert_called_once()
        # Assert that subprocess.run was *not* called because the write failed first
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_update_persistent_library_invalid_path(
    mocker,
):  # Keep mocker for subprocess mock
    item = KBItem(unique_name="Test.InvalidPath.Item", lean_code="def z := 3")
    invalid_path = pathlib.Path("/non/existent/path")  # Path that doesn't exist
    lake_exe = "lake"
    timeout = 30
    # Mock run, although it shouldn't be called due to the path check
    mock_run = mocker.patch("subprocess.run")
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
    # Mock external processes and file system interactions using mocker
    mocker.patch("subprocess.run", autospec=True)
    mocker.patch("os.makedirs", autospec=True)
    mocker.patch(
        "shutil.which", return_value="/path/to/lean"
    )  # Assume lean is findable

    # Mock environment variable access if needed explicitly beyond setup
    mocker.patch(
        "os.getenv", return_value=None
    )  # Default mock, override in tests if needed

    # Mock tempfile.TemporaryDirectory context manager correctly
    mock_tmp_dir_instance = MagicMock(spec=tempfile.TemporaryDirectory)
    mock_tmp_dir_instance.name = "/fake/temp/dir/pytest-tmp"  # Use a fixed fake path
    mock_tmp_dir_instance.__enter__.return_value = mock_tmp_dir_instance.name
    mock_tmp_dir_instance.__exit__.return_value = None

    # Mock the cleanup method explicitly if called directly
    # Use mocker to patch the method on the class
    mocker.patch.object(tempfile.TemporaryDirectory, "cleanup", autospec=True)

    # Configure the class mock to return the instance when called
    mock_tmp_dir_class = mocker.patch("tempfile.TemporaryDirectory", autospec=True)
    mock_tmp_dir_class.return_value = (
        mock_tmp_dir_instance  # The call returns our instance
    )

    # Mock loggers to prevent console noise and allow assertion if needed
    mocker.patch.object(lean_interaction_logger, "info")
    mocker.patch.object(lean_interaction_logger, "warning")
    mocker.patch.object(lean_interaction_logger, "error")
    mocker.patch.object(lean_interaction_logger, "exception")
    mocker.patch.object(lean_interaction_logger, "debug")


@pytest.mark.asyncio
async def test_check_and_compile_item_not_found(mocker, tmp_path):  # Keep mocker
    unique_name = "Non.Existent.Item"
    # Patch DB function where it is looked up within lean_interaction's scope
    # using mocker
    mock_get_item = mocker.patch.object(
        lean_interaction_module, "get_kb_item_by_name", return_value=None
    )
    mock_save_item = mocker.patch.object(
        lean_interaction_module, "save_kb_item", new_callable=AsyncMock
    )

    # Mock Helper functions LOCALLY (though they shouldn't be called) using mocker
    mock_create_temp = mocker.patch.object(
        lean_interaction_module, "_create_temp_env_for_verification"
    )
    mock_update_persist = mocker.patch.object(
        lean_interaction_module, "_update_persistent_library", new_callable=AsyncMock
    )
    # subprocess.run is mocked by the patch_core_dependencies fixture

    # Ensure SHARED_LIB_PATH is valid for the initial check in check_and_compile_item
    shared_path_obj = tmp_path / "shared_valid_for_check"
    shared_path_obj.mkdir()
    # Patch the resolved SHARED_LIB_PATH constant within the module using mocker
    mocker.patch.object(lean_interaction_module, "SHARED_LIB_PATH", shared_path_obj)

    success, message = await check_and_compile_item(unique_name)

    assert success is False
    assert f"Target item '{unique_name}' not found" in message, (
        f"Message was: {message}"
    )
    mock_get_item.assert_called_once_with(unique_name, db_path=DEFAULT_DB_PATH)
    mock_save_item.assert_not_called()
    mock_create_temp.assert_not_called()
    # Check subprocess mock (mocked by fixture)
    # Retrieve the mock created by the fixture to assert
    subprocess_run_mock = subprocess.run
    subprocess_run_mock.assert_not_called()
    mock_update_persist.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_compile_item_config_error_no_shared_lib(mocker):  # Keep mocker
    unique_name = "Any.Item"
    # Patch DB functions (should not be called), targeting lean_interaction using mocker
    mock_get_item = mocker.patch.object(lean_interaction_module, "get_kb_item_by_name")
    mock_save_item = mocker.patch.object(
        lean_interaction_module, "save_kb_item", new_callable=AsyncMock
    )

    # Patch the resolved SHARED_LIB_PATH constant to be None within the module
    # using mocker
    mocker.patch.object(lean_interaction_module, "SHARED_LIB_PATH", None)

    success, message = await check_and_compile_item(unique_name)

    assert success is False
    # Check the specific error message generated when SHARED_LIB_PATH is None/invalid
    assert (
        "Shared library path (LEAN_AUTOMATOR_SHARED_LIB_PATH) not configured correctly"
        in message
    ), f"Message was: {message}"
    mock_get_item.assert_not_called()
    mock_save_item.assert_not_called()


# Note: Additional tests for check_and_compile_item covering success,
# temp build failure, persistent update failure etc. would go here,
# ensuring mocks for get_kb_item_by_name, save_kb_item, _create_temp_env...,
# _update_persistent_library, and subprocess.run are set up appropriately
# for each scenario using either patch.object or mocker.patch/mocker.patch.object.
# Remember to patch SHARED_LIB_PATH to a valid mock path using
# patch.object or mocker.patch.object for tests that should pass the initial
# configuration check.
