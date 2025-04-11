# tests/unit/lean/test_interaction_unit.py

import pathlib
import subprocess
import sys
from unittest.mock import ANY, AsyncMock, MagicMock  # Removed patch

import pytest

# --- Determine project root dynamically ---
# Assuming this script is in tests/unit/lean/
current_dir = pathlib.Path(__file__).parent
# Navigate up: lean -> unit -> tests -> project root
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))  # Add src to path

# --- Imports for module under test and dependencies ---
try:
    # Import the module under test
    # Import necessary components from kb.storage
    from lean_automator.kb.storage import (
        DEFAULT_DB_PATH,
        ItemStatus,
        ItemType,
        KBItem,
    )
    from lean_automator.lean import interaction as lean_interaction_module

    # Import specific items to be tested or mocked
    from lean_automator.lean.interaction import (
        LeanVerifier,  # Import the class itself
        _module_name_to_path,
        _write_text_sync,  # Internal helper used via executor
        check_and_compile_item,
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


@pytest.fixture
def mock_kb_item():
    """Provides a basic mock KBItem."""
    return KBItem(
        id=1,
        unique_name="VantageLib.Test.Lib.SampleItem",
        item_type=ItemType.THEOREM,
        description_nl="A test item.",
        lean_code="theorem foo : True := by trivial",
        topic="Testing",
        status=ItemStatus.PENDING,
        plan_dependencies=["VantageLib.Test.Lib.Dep1", "VantageLib.Test.Lib.Dep2"],
    )


# --- Unit Tests for Helper Functions ---


@pytest.mark.parametrize(
    "module_name, expected_path_parts",
    [
        ("MyModule", ("MyModule",)),
        ("Category.Theory.Limits", ("Category", "Theory", "Limits")),
    ],
)
def test_module_name_to_path_valid(module_name, expected_path_parts):
    expected = pathlib.Path(*expected_path_parts)
    assert _module_name_to_path(module_name) == expected


@pytest.mark.parametrize("invalid_module_name", ["", ".", "..", "..."])
def test_module_name_to_path_invalid(invalid_module_name):
    with pytest.raises(ValueError):
        _module_name_to_path(invalid_module_name)


@pytest.mark.asyncio
async def test_generate_imports_for_target_with_deps(mocker):
    item = KBItem(
        unique_name="VantageLib.Test.Module.Item",
        plan_dependencies=[
            "VantageLib.MyLib.Dep.One",
            "VantageLib.MyLib.Dep.Two.Sub",
            "VantageLib.Test.Module.Item",
        ],
        item_type=ItemType.THEOREM,
        description_nl="Desc",
        lean_code="code",
        topic="Topic",
    )
    mock_shared_path = pathlib.Path("/fake/valid/shared/lib")
    mocker.patch.object(
        LeanVerifier, "_resolve_shared_lib_path", return_value=mock_shared_path
    )
    mocker.patch(
        "lean_automator.lean.interaction.get_lean_automator_shared_lib_path",
        return_value=str(mock_shared_path),
    )
    mocker.patch(
        "lean_automator.lean.interaction.APP_CONFIG",
        {
            "lean_paths": {
                "shared_lib_package_name": "VantageLib",
                "shared_lib_src_dir_name": "VantageLib",
            }
        },
    )

    verifier = LeanVerifier(shared_lib_path_str="/fake/lib")

    expected_imports = (
        "-- Auto-generated imports based on plan_dependencies --\n"
        "import VantageLib.MyLib.Dep.One\n"
        "import VantageLib.MyLib.Dep.Two.Sub"
    )
    assert verifier._generate_imports_for_target(item) == expected_imports


@pytest.mark.asyncio
async def test_generate_imports_for_target_no_deps(mocker):
    item = KBItem(
        unique_name="VantageLib.Another.Item",
        plan_dependencies=[],
        item_type=ItemType.THEOREM,
        description_nl="Desc",
        lean_code="code",
        topic="Topic",
    )
    mock_shared_path = pathlib.Path("/fake/valid/shared/lib")
    mocker.patch.object(
        LeanVerifier, "_resolve_shared_lib_path", return_value=mock_shared_path
    )
    mocker.patch(
        "lean_automator.lean.interaction.get_lean_automator_shared_lib_path",
        return_value=str(mock_shared_path),
    )
    mocker.patch(
        "lean_automator.lean.interaction.APP_CONFIG",
        {
            "lean_paths": {
                "shared_lib_package_name": "VantageLib",
                "shared_lib_src_dir_name": "VantageLib",
            }
        },
    )

    verifier = LeanVerifier(shared_lib_path_str="/fake/lib")
    assert verifier._generate_imports_for_target(item) == ""


@pytest.mark.asyncio
async def test_generate_imports_for_target_already_prefixed(mocker):
    item = KBItem(
        unique_name="VantageLib.My.Item",
        plan_dependencies=["VantageLib.Dep.One", "VantageLib.Dep.Two"],
        item_type=ItemType.THEOREM,
        description_nl="Desc",
        lean_code="code",
        topic="Topic",
    )
    mock_shared_path = pathlib.Path("/fake/valid/shared/lib")
    mocker.patch.object(
        LeanVerifier, "_resolve_shared_lib_path", return_value=mock_shared_path
    )
    mocker.patch(
        "lean_automator.lean.interaction.get_lean_automator_shared_lib_path",
        return_value=str(mock_shared_path),
    )
    mocker.patch(
        "lean_automator.lean.interaction.APP_CONFIG",
        {
            "lean_paths": {
                "shared_lib_package_name": "VantageLib",
                "shared_lib_src_dir_name": "VantageLib",
            }
        },
    )

    verifier = LeanVerifier(shared_lib_path_str="/fake/lib")

    expected_imports = (
        "-- Auto-generated imports based on plan_dependencies --\n"
        "import VantageLib.Dep.One\n"
        "import VantageLib.Dep.Two"
    )
    assert verifier._generate_imports_for_target(item) == expected_imports


def test_write_text_sync(tmp_path):
    """Test the synchronous file writing helper."""
    file_path = tmp_path / "subdir" / "test_write.txt"
    content = "Hello\nWorld!"
    _write_text_sync(file_path, content)
    assert file_path.read_text(encoding="utf-8") == content
    assert file_path.parent.is_dir()


# --- Unit Tests for check_and_compile_item (Testing LeanVerifier implicitly) ---


@pytest.fixture(autouse=True)
def patch_verifier_dependencies(mocker, mock_subprocess_result):
    """Auto-used fixture to mock external dependencies for LeanVerifier methods."""
    mock_get = mocker.patch(
        "lean_automator.lean.interaction.get_kb_item_by_name", return_value=None
    )
    mock_save = mocker.patch(
        "lean_automator.lean.interaction.save_kb_item", new_callable=AsyncMock
    )
    # Mock subprocess.run globally first
    mock_run = mocker.patch("subprocess.run", return_value=mock_subprocess_result())

    # More specific mocking for subprocess.run if needed, e.g., for --print-libdir
    # If the default mock_subprocess_result (code 0, empty output) is fine for
    # the --print-libdir call, we don't need extra side_effect logic here.
    # If we needed --print-libdir to return something specific, we'd use side_effect:
    # def run_side_effect(*args, **kwargs):
    #     cmd_args = args[0] if args else kwargs.get('args', [])
    #     if cmd_args == ['/fake/path/to/lean', '--print-libdir']:
    #         return mock_subprocess_result(stdout="/fake/stdlib/path")
    #     elif cmd_args[0] == 'lake':
    #         # Return default or allow test-specific configuration
    #         # Be careful here, test specific side_effects might override this
    #         return mock_subprocess_result()
    #     return mock_subprocess_result() # Default fallback
    # mock_run.side_effect = run_side_effect

    mocker.patch("os.makedirs")
    mock_remove = mocker.patch("os.remove")
    mock_write_sync = mocker.patch("lean_automator.lean.interaction._write_text_sync")
    mocker.patch("shutil.which", return_value="/fake/path/to/lean")
    mocker.patch("os.getenv", return_value=None)

    mock_analyze = mocker.patch(
        "lean_automator.lean.interaction.analyze_lean_failure",
        new_callable=AsyncMock,
        return_value="LSP Analysis Result",
    )

    mock_shared_path = pathlib.Path("/fake/valid/shared/lib")
    mocker.patch.object(
        lean_interaction_module.LeanVerifier,
        "_resolve_shared_lib_path",
        return_value=mock_shared_path,
    )
    mocker.patch(
        "lean_automator.lean.interaction.get_lean_automator_shared_lib_path",
        return_value=str(mock_shared_path),
    )
    mocker.patch(
        "lean_automator.lean.interaction.APP_CONFIG",
        {
            "lean_paths": {
                "shared_lib_package_name": "VantageLib",
                "shared_lib_src_dir_name": "VantageLib",
            }
        },
    )

    return {
        "get_item": mock_get,
        "save_item": mock_save,
        "subprocess_run": mock_run,  # Return the main mock object
        "os_remove": mock_remove,
        "write_sync": mock_write_sync,
        "analyze_failure": mock_analyze,
    }


@pytest.mark.asyncio
async def test_check_and_compile_item_success(
    patch_verifier_dependencies, mock_kb_item
):
    mock_get_item = patch_verifier_dependencies["get_item"]
    mock_save_item = patch_verifier_dependencies["save_item"]
    mock_subprocess_run = patch_verifier_dependencies["subprocess_run"]
    mock_os_remove = patch_verifier_dependencies["os_remove"]
    mock_write_sync = patch_verifier_dependencies["write_sync"]

    mock_get_item.return_value = mock_kb_item
    # Ensure the build call returns success (default from fixture is already success)
    # mock_subprocess_run.return_value = mock_subprocess_result(returncode=0)
    # Already default

    success, message = await check_and_compile_item(mock_kb_item.unique_name)

    assert success is True
    assert "verified and integrated" in message
    mock_get_item.assert_called_once_with(
        mock_kb_item.unique_name, db_path=DEFAULT_DB_PATH
    )
    mock_write_sync.assert_called_once()

    # Assert that the 'lake build' command was attempted
    expected_build_args = ["lake", "build", "VantageLib.Test.Lib.SampleItem"]
    mock_subprocess_run.assert_any_call(
        args=expected_build_args,
        cwd="/fake/valid/shared/lib",  # Check CWD
        env=ANY,  # Ignore complex env dict for this check
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    mock_save_item.assert_called_once()
    saved_item = mock_save_item.call_args[0][0]
    assert saved_item.status == ItemStatus.PROVEN
    assert saved_item.lean_error_log is None
    mock_os_remove.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_compile_item_build_fails(
    patch_verifier_dependencies, mock_kb_item, mock_subprocess_result
):
    mock_get_item = patch_verifier_dependencies["get_item"]
    mock_save_item = patch_verifier_dependencies["save_item"]
    mock_subprocess_run = patch_verifier_dependencies["subprocess_run"]
    mock_os_remove = patch_verifier_dependencies["os_remove"]
    mock_write_sync = patch_verifier_dependencies["write_sync"]
    mock_analyze_failure = patch_verifier_dependencies["analyze_failure"]

    mock_get_item.return_value = mock_kb_item
    # Configure the *main* mock to return failure for the build call
    # If using side_effect in fixture, configure it there.
    # Otherwise, set return_value directly.
    mock_subprocess_run.return_value = mock_subprocess_result(
        returncode=1, stderr="Lean build error!"
    )
    mock_analyze_failure.return_value = "Detailed LSP Log"

    success, message = await check_and_compile_item(mock_kb_item.unique_name)

    assert success is False
    assert "Lean validation failed" in message
    assert "File removed" in message
    mock_get_item.assert_called_once()
    mock_write_sync.assert_called_once()

    # Assert that the 'lake build' command was attempted
    expected_build_args = ["lake", "build", "VantageLib.Test.Lib.SampleItem"]
    mock_subprocess_run.assert_any_call(
        args=expected_build_args,
        cwd="/fake/valid/shared/lib",
        env=ANY,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    mock_analyze_failure.assert_called_once()
    analyze_call_args, analyze_call_kwargs = mock_analyze_failure.call_args
    assert analyze_call_kwargs.get("cwd") == "/fake/valid/shared/lib"

    mock_save_item.assert_called_once()
    saved_item = mock_save_item.call_args[0][0]
    assert saved_item.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert saved_item.lean_error_log == "Detailed LSP Log"
    assert getattr(saved_item, "failure_count", 0) == 1
    mock_os_remove.assert_called_once()


@pytest.mark.asyncio
async def test_check_and_compile_item_write_fails(
    patch_verifier_dependencies, mock_kb_item
):
    mock_get_item = patch_verifier_dependencies["get_item"]
    mock_save_item = patch_verifier_dependencies["save_item"]
    mock_subprocess_run = patch_verifier_dependencies["subprocess_run"]
    mock_os_remove = patch_verifier_dependencies["os_remove"]
    mock_write_sync = patch_verifier_dependencies["write_sync"]
    mock_analyze_failure = patch_verifier_dependencies["analyze_failure"]

    mock_get_item.return_value = mock_kb_item
    mock_write_sync.side_effect = OSError("Disk full!")

    success, message = await check_and_compile_item(mock_kb_item.unique_name)

    assert success is False
    assert "File system error writing" in message
    assert "Disk full!" in message
    mock_get_item.assert_called_once()
    mock_write_sync.assert_called_once()
    # Subprocess should NOT have been called if write failed
    mock_subprocess_run.assert_not_called()
    mock_analyze_failure.assert_not_called()
    mock_save_item.assert_called_once()
    saved_item = mock_save_item.call_args[0][0]
    assert saved_item.status == ItemStatus.ERROR
    assert "Disk full!" in saved_item.lean_error_log
    mock_os_remove.assert_not_called()


@pytest.mark.asyncio
async def test_check_and_compile_item_timeout(
    patch_verifier_dependencies, mock_kb_item
):
    mock_get_item = patch_verifier_dependencies["get_item"]
    mock_save_item = patch_verifier_dependencies["save_item"]
    mock_subprocess_run = patch_verifier_dependencies["subprocess_run"]
    mock_os_remove = patch_verifier_dependencies["os_remove"]
    mock_write_sync = patch_verifier_dependencies["write_sync"]
    mock_analyze_failure = patch_verifier_dependencies["analyze_failure"]

    mock_get_item.return_value = mock_kb_item
    # Configure the mock to raise TimeoutExpired when called
    mock_subprocess_run.side_effect = subprocess.TimeoutExpired(
        cmd="mock lake build", timeout=10
    )

    success, message = await check_and_compile_item(mock_kb_item.unique_name)

    assert success is False
    assert "Build execution error" in message or "Timeout" in message

    mock_get_item.assert_called_once()
    mock_write_sync.assert_called_once()

    expected_build_args = ["lake", "build", "VantageLib.Test.Lib.SampleItem"]
    mock_subprocess_run.assert_any_call(
        args=expected_build_args,
        cwd="/fake/valid/shared/lib",
        env=ANY,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    mock_analyze_failure.assert_not_called()
    mock_save_item.assert_called_once()
    saved_item = mock_save_item.call_args[0][0]
    assert saved_item.status == ItemStatus.ERROR
    # Corrected assertion below:
    assert (
        "timed out after" in saved_item.lean_error_log
    )  # Check for content that is actually logged

    mock_os_remove.assert_called_once()


@pytest.mark.asyncio
async def test_check_and_compile_item_lsp_analysis_fails(
    patch_verifier_dependencies, mock_kb_item, mock_subprocess_result
):
    mock_get_item = patch_verifier_dependencies["get_item"]
    mock_save_item = patch_verifier_dependencies["save_item"]
    mock_subprocess_run = patch_verifier_dependencies["subprocess_run"]
    mock_os_remove = patch_verifier_dependencies["os_remove"]
    mock_write_sync = patch_verifier_dependencies["write_sync"]
    mock_analyze_failure = patch_verifier_dependencies["analyze_failure"]

    mock_get_item.return_value = mock_kb_item
    # Configure subprocess mock to return failure
    mock_subprocess_run.return_value = mock_subprocess_result(
        returncode=1, stderr="Lean build error!"
    )
    # Configure analyze mock to raise an error
    mock_analyze_failure.side_effect = Exception("LSP server crashed!")

    success, message = await check_and_compile_item(mock_kb_item.unique_name)

    assert success is False
    assert "Lean validation failed" in message
    mock_get_item.assert_called_once()
    mock_write_sync.assert_called_once()

    # Assert that the 'lake build' command was attempted
    expected_build_args = ["lake", "build", "VantageLib.Test.Lib.SampleItem"]
    mock_subprocess_run.assert_any_call(
        args=expected_build_args,
        cwd="/fake/valid/shared/lib",
        env=ANY,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    mock_analyze_failure.assert_called_once()  # LSP was attempted
    mock_save_item.assert_called_once()
    saved_item = mock_save_item.call_args[0][0]
    assert saved_item.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert "LSP Analysis Failed: LSP server crashed!" in saved_item.lean_error_log
    assert getattr(saved_item, "failure_count", 0) == 1
    mock_os_remove.assert_called_once()


@pytest.mark.asyncio
async def test_check_and_compile_item_config_error(mocker):
    mock_get_item = mocker.patch("lean_automator.lean.interaction.get_kb_item_by_name")
    # Patch the helper method on the class itself that __init__ calls
    mocker.patch.object(
        lean_interaction_module.LeanVerifier,
        "_resolve_shared_lib_path",
        return_value=None,
    )
    mocker.patch(
        "lean_automator.lean.interaction.get_lean_automator_shared_lib_path",
        return_value=None,
    )
    mocker.patch("lean_automator.lean.interaction.APP_CONFIG", {})

    success, message = await check_and_compile_item("Any.Item")

    assert success is False
    assert "Configuration error" in message
    assert "Shared library path" in message
    mock_get_item.assert_not_called()
