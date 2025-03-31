# File: tests/unit/test_lean_interaction_unit.py

import pytest
import pytest_asyncio
import pathlib
import sys
import os
import subprocess
import tempfile
import json
import logging
import sqlite3
import builtins
from unittest.mock import patch, MagicMock, call, ANY, AsyncMock
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, Optional, List, Any
import time
import asyncio
import shutil
import copy

# --- Test Setup ---
# Ensure imports work relative to the project root
# (No changes needed here if project structure allows imports as before)

try:
    from src.lean_automator.lean_interaction import (
        _module_name_to_path,
        _fetch_recursive_dependencies,
        _create_temp_env_lake,
        check_and_compile_item,
        logger as lean_interaction_logger
    )
    from src.lean_automator.kb_storage import (
         KBItem, ItemStatus, ItemType, LatexLink, _sentinel, DEFAULT_DB_PATH,
         save_kb_item,
         get_kb_item_by_name,
    )
    # Import the specific module for patching isinstance inside it
    from src.lean_automator import kb_storage as kb_storage_module
except ImportError as e:
    pytest.skip(f"Skipping lean_interaction unit tests due to import error: {e}", allow_module_level=True)

# --- Helper & Fixture ---

def mock_completed_process(returncode=0, stdout="", stderr=""):
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc

@pytest.fixture(scope="function")
def mock_kb_items():
    # No changes needed to the test data itself
    now = datetime.now(timezone.utc)
    items = {}
    items["Lib.A"] = KBItem(
            unique_name="Lib.A", id=1, item_type=ItemType.DEFINITION,
            lean_code="def A_val : Nat := 1", status=ItemStatus.PROVEN,
            lean_olean=b'olean_data_A', plan_dependencies=[], created_at=now, last_modified_at=now
        )
    now += timedelta(milliseconds=1)
    items["Lib.B"] = KBItem(
            unique_name="Lib.B", id=2, item_type=ItemType.THEOREM,
            lean_code="import Lib.A\ntheorem B_thm : A_val = 1 := rfl", status=ItemStatus.PENDING,
            plan_dependencies=["Lib.A"], created_at=now, last_modified_at=now
        )
    now += timedelta(milliseconds=1)
    items["Lib.C"] = KBItem(
            unique_name="Lib.C", id=3, item_type=ItemType.DEFINITION,
            lean_code="def C_val : Nat := 30", status=ItemStatus.PENDING,
            plan_dependencies=[], created_at=now, last_modified_at=now
        )
    now += timedelta(milliseconds=1)
    items["Lib.D"] = KBItem(
            unique_name="Lib.D", id=4, item_type=ItemType.THEOREM,
            lean_code="import Lib.C\ntheorem D_thm : C_val = 30 := rfl", status=ItemStatus.PENDING,
            plan_dependencies=["Lib.C"], created_at=now, last_modified_at=now
        )
    now += timedelta(milliseconds=1)
    items["Lib.E_BadDep"] = KBItem(
            unique_name="Lib.E_BadDep", id=5, item_type=ItemType.DEFINITION,
            lean_code="def E_val : Nat := 5", status=ItemStatus.PROVEN,
            lean_olean=None,
            plan_dependencies=[], created_at=now, last_modified_at=now
        )
    now += timedelta(milliseconds=1)
    items["Lib.F_UsesBadDep"] = KBItem(
            unique_name="Lib.F_UsesBadDep", id=6, item_type=ItemType.THEOREM,
            lean_code="import Lib.E_BadDep\ntheorem F_thm : E_val = 5 := rfl", status=ItemStatus.PENDING,
            plan_dependencies=["Lib.E_BadDep"], created_at=now, last_modified_at=now
        )
    now += timedelta(milliseconds=1)
    items["CycleX"] = KBItem(unique_name="CycleX", id=7, item_type=ItemType.THEOREM, lean_code="import CycleY", plan_dependencies=["CycleY"], created_at=now, last_modified_at=now)
    now += timedelta(milliseconds=1)
    items["CycleY"] = KBItem(unique_name="CycleY", id=8, item_type=ItemType.THEOREM, lean_code="import CycleX", plan_dependencies=["CycleX"], created_at=now, last_modified_at=now)
    now += timedelta(milliseconds=1)
    items["ItemZ_MissingDep"] = KBItem(unique_name="ItemZ_MissingDep", id=9, item_type=ItemType.THEOREM, lean_code="def Z := 0", plan_dependencies=["MissingDep"], created_at=now, last_modified_at=now)
    now += timedelta(milliseconds=1)
    items["RemarkOnly"] = KBItem(unique_name="RemarkOnly", id=10, item_type=ItemType.REMARK, lean_code="", status=ItemStatus.PENDING, plan_dependencies=[], created_at=now, last_modified_at=now)
    return items


# --- Unit Tests for _module_name_to_path ---
@pytest.mark.parametrize(
    "module_name, expected_parts",
    [
        ("Basics", ["Basics"]),
        ("Category.Theory", ["Category", "Theory"]),
        ("Data.List.Basic", ["Data", "List", "Basic"]),
        ("SingleWord", ["SingleWord"]),
        ("A.B", ["A", "B"]),
    ]
)
def test_module_name_to_path_valid(module_name, expected_parts):
    expected_path = pathlib.Path(*expected_parts)
    actual_path = _module_name_to_path(module_name)
    assert actual_path == expected_path
    assert isinstance(actual_path, pathlib.Path)

@pytest.mark.parametrize(
    "invalid_name", ["", ".", ".."]
)
def test_module_name_to_path_invalid_empty(invalid_name):
    with pytest.raises(ValueError, match="Invalid module name format"):
        _module_name_to_path(invalid_name)

def test_module_name_to_path_handles_double_dots():
    assert _module_name_to_path("A..B") == pathlib.Path("A", "B")


# --- Unit Tests for _fetch_recursive_dependencies ---
@patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
class TestFetchRecursiveDependencies:

    def test_fetch_recursive_dependencies_single_item(self, mock_get_item, mock_kb_items):
        target_name = "Lib.A"
        mock_get_item.return_value = mock_kb_items.get(target_name)
        visited_names: Set[str] = set()
        fetched_items: Dict[str, KBItem] = {}
        _fetch_recursive_dependencies(target_name, "/tmp/fake_db1", visited_names, fetched_items)
        assert target_name in fetched_items
        assert len(fetched_items) == 1
        mock_get_item.assert_called_once_with(target_name, db_path="/tmp/fake_db1")

    def test_fetch_recursive_dependencies_simple_chain(self, mock_get_item, mock_kb_items):
        target_name = "Lib.B"
        def side_effect_func(name, db_path=None):
            return mock_kb_items.get(name)
        mock_get_item.side_effect = side_effect_func
        visited_names: Set[str] = set()
        fetched_items: Dict[str, KBItem] = {}
        _fetch_recursive_dependencies(target_name, "/tmp/fake_db2", visited_names, fetched_items)
        assert "Lib.A" in fetched_items
        assert "Lib.B" in fetched_items
        assert len(fetched_items) == 2
        mock_get_item.assert_has_calls([
            call("Lib.B", db_path="/tmp/fake_db2"),
            call("Lib.A", db_path="/tmp/fake_db2"),
        ], any_order=True)
        assert mock_get_item.call_count == 2

    def test_fetch_recursive_dependencies_cycle(self, mock_get_item, mocker, mock_kb_items):
        target_name = "CycleX"
        mock_log_warning = mocker.patch.object(lean_interaction_logger, 'warning')
        def side_effect_func(name, db_path=None):
            return mock_kb_items.get(name)
        mock_get_item.side_effect = side_effect_func
        visited_names: Set[str] = set()
        fetched_items: Dict[str, KBItem] = {}
        _fetch_recursive_dependencies(target_name, "/tmp/fake_db3", visited_names, fetched_items)
        assert "CycleX" in fetched_items
        assert "CycleY" in fetched_items
        assert len(fetched_items) == 2
        found_cycle_warning = False
        for call_args in mock_log_warning.call_args_list:
            if "Cycle detected" in call_args[0][0] or "Skipping already visited" in call_args[0][0]:
                 found_cycle_warning = True
                 break
        assert found_cycle_warning is False

    def test_fetch_recursive_dependencies_missing(self, mock_get_item, mock_kb_items):
        target_name = "ItemZ_MissingDep"
        missing_dep_name = "MissingDep"
        def side_effect_func(name, db_path=None):
            return mock_kb_items.get(name)
        mock_get_item.side_effect = side_effect_func
        visited_names: Set[str] = set()
        fetched_items: Dict[str, KBItem] = {}
        with pytest.raises(FileNotFoundError, match=f"Item '{missing_dep_name}' not found"):
            _fetch_recursive_dependencies(target_name, "/tmp/fake_db4", visited_names, fetched_items)
        assert "ItemZ_MissingDep" in fetched_items
        assert mock_get_item.call_count == 2


# --- Unit Tests for _create_temp_env_lake ---
@pytest.mark.usefixtures("fs")
class TestCreateTempEnvLake:

    def test_create_temp_env_lake_simple(self, fs):
        lib_name = "MyLib"
        target_item = KBItem(unique_name="Target", lean_code="def T := 1")
        all_items = {"Target": target_item}
        temp_dir_base = "/tmp/pytest_fakefs/proj"
        fs.create_dir(temp_dir_base)
        proj_path_str, target_module_name, target_olean_path_str = _create_temp_env_lake(
            target_item, all_items, temp_dir_base, lib_name=lib_name
        )
        assert proj_path_str == temp_dir_base
        assert fs.exists(pathlib.Path(temp_dir_base) / "lakefile.lean")
        assert fs.exists(pathlib.Path(temp_dir_base) / lib_name / "Target.lean")
        expected_olean = pathlib.Path(temp_dir_base) / ".lake" / "build" / "lib" / lib_name / "Target.olean"
        assert target_olean_path_str == str(expected_olean)

    def test_create_temp_env_lake_with_verified_dep(self, fs, mock_kb_items, mocker):
        lib_name = "MyLib"
        target_item = mock_kb_items["Lib.B"]
        dep_item = mock_kb_items["Lib.A"]
        all_items = {"Lib.B": target_item, "Lib.A": dep_item}
        temp_dir_base = "/tmp/pytest_fakefs/proj_dep"
        fs.create_dir(temp_dir_base)
        dep_olean_path = pathlib.Path(temp_dir_base) / ".lake" / "build" / "lib" / lib_name / "Lib" / "A.olean"
        proj_path_str, target_module_name, target_olean_path_str = _create_temp_env_lake(
            target_item, all_items, temp_dir_base, lib_name=lib_name
        )
        assert fs.exists(pathlib.Path(temp_dir_base) / lib_name / "Lib" / "B.lean")
        assert fs.exists(pathlib.Path(temp_dir_base) / lib_name / "Lib" / "A.lean")
        assert not fs.exists(dep_olean_path)

    def test_create_temp_env_lake_with_initialized_dep(self, fs, mock_kb_items):
        lib_name = "MyLib"
        target_item = mock_kb_items["Lib.D"]
        dep_item = mock_kb_items["Lib.C"]
        all_items = {"Lib.D": target_item, "Lib.C": dep_item}
        temp_dir_base = "/tmp/pytest_fakefs/proj_init"
        fs.create_dir(temp_dir_base)
        _create_temp_env_lake(target_item, all_items, temp_dir_base, lib_name=lib_name)
        assert fs.exists(pathlib.Path(temp_dir_base) / lib_name / "Lib" / "D.lean")
        assert fs.exists(pathlib.Path(temp_dir_base) / lib_name / "Lib" / "C.lean")
        dep_olean_path = pathlib.Path(temp_dir_base) / ".lake" / "build" / "lib" / lib_name / "Lib" / "C.olean"
        assert not fs.exists(dep_olean_path)

    def test_create_temp_env_lake_dep_verified_missing_olean(self, fs, mock_kb_items):
        lib_name = "MyLib"
        target_item = mock_kb_items["Lib.F_UsesBadDep"]
        dep_item = mock_kb_items["Lib.E_BadDep"]
        all_items = {"Lib.F_UsesBadDep": target_item, "Lib.E_BadDep": dep_item}
        temp_dir_base = "/tmp/pytest_fakefs/proj_bad"
        fs.create_dir(temp_dir_base)
        _create_temp_env_lake(target_item, all_items, temp_dir_base, lib_name=lib_name)
        assert fs.exists(pathlib.Path(temp_dir_base) / lib_name / "Lib" / "F_UsesBadDep.lean")
        assert fs.exists(pathlib.Path(temp_dir_base) / lib_name / "Lib" / "E_BadDep.lean")
        dep_olean_path = pathlib.Path(temp_dir_base) / ".lake" / "build" / "lib" / lib_name / "Lib" / "E_BadDep.olean"
        assert not fs.exists(dep_olean_path)

    def test_create_temp_env_lake_item_no_code(self, fs, mock_kb_items):
        lib_name = "MyLib"
        target_item = mock_kb_items["RemarkOnly"]
        all_items = {"RemarkOnly": target_item}
        temp_dir_base = "/tmp/pytest_fakefs/proj_remark"
        fs.create_dir(temp_dir_base)
        proj_path_str, target_module_name, target_olean_path_str = _create_temp_env_lake(
            target_item, all_items, temp_dir_base, lib_name=lib_name
        )
        assert fs.exists(pathlib.Path(temp_dir_base) / "lakefile.lean")
        assert not fs.exists(pathlib.Path(temp_dir_base) / lib_name / "RemarkOnly.lean")
        expected_olean = pathlib.Path(temp_dir_base) / ".lake" / "build" / "lib" / lib_name / "RemarkOnly.olean"
        assert target_olean_path_str == str(expected_olean)
        assert proj_path_str == temp_dir_base
        assert target_module_name == f"{lib_name}.RemarkOnly"


# --- Unit Tests for check_and_compile_item (Updated) ---

# Helper to setup mocks for subprocess.run and os/pathlib related to LAKE_HOME
def configure_mocks_for_compile(mocker, lake_cache_path=None, cache_dir_exists=True, cache_creation_error=None, lean_libdir="/fake/lean/libdir", lean_libdir_error=None, lake_returncode=0, lake_stdout="", lake_stderr="", lake_exception=None):
    """Configures mocks for subprocess.run, os.getenv, pathlib.Path.is_dir, os.makedirs."""

    mocker.patch.dict(os.environ, {'LEAN_AUTOMATOR_LAKE_CACHE': lake_cache_path} if lake_cache_path else {}, clear=True)

    # Patch pathlib.Path methods related to cache check
    # We patch the *class* methods, and the side_effect will receive the instance
    mock_path_exists = mocker.patch('pathlib.Path.exists', autospec=True)
    mock_path_is_dir = mocker.patch('pathlib.Path.is_dir', autospec=True)
    mock_path_resolve = mocker.patch('pathlib.Path.resolve', autospec=True)

    def path_resolve_side_effect(self):
        # Return a resolved version, potentially mock this more if needed
        return self

    def path_exists_side_effect(self):
        # Generally assume paths exist unless it's the specific cache path we are testing
        if lake_cache_path and str(self.resolve()) == str(pathlib.Path(lake_cache_path).resolve()):
            return cache_dir_exists
        return True # Assume other paths (like temp build dir) exist

    def path_is_dir_side_effect(self):
        if lake_cache_path and str(self.resolve()) == str(pathlib.Path(lake_cache_path).resolve()):
            return cache_dir_exists
        # Make a reasonable assumption for other paths if necessary, e.g., temp dir
        if str(self).startswith("/tmp"): # Crude check for temp paths
             return True
        return False # Default for other paths

    mock_path_resolve.side_effect = path_resolve_side_effect
    # Apply the is_dir side effect ONLY if a cache path is involved in the test
    if lake_cache_path:
        mock_path_is_dir.side_effect = path_is_dir_side_effect
    # exists() might also be called, mock it generally or specifically
    # For now, let's assume relevant paths exist unless it's the cache being created
    # mocker.patch('pathlib.Path.exists').return_value = True # Too broad?

    # Mock os.makedirs
    mock_makedirs = mocker.patch('os.makedirs')
    if lake_cache_path and not cache_dir_exists:
        if cache_creation_error:
            mock_makedirs.side_effect = cache_creation_error
        else:
            def makedirs_side_effect(path, exist_ok=False):
                # Check if called with the resolved cache path
                abs_path_arg = pathlib.Path(path).resolve()
                expected_path = pathlib.Path(lake_cache_path).resolve()
                if str(abs_path_arg) != str(expected_path):
                     # Allow creation of parent dirs for lean files in temp env
                     if not str(abs_path_arg).startswith(str(mocker.MagicMock().name)): # Hacky check if it's the temp dir
                        raise ValueError(f"os.makedirs called with unexpected path: {abs_path_arg} vs {expected_path}")
                 # Simulate successful creation
            mock_makedirs.side_effect = makedirs_side_effect


    # Mock subprocess.run remains mostly the same, verifying LAKE_HOME in env
    def subprocess_side_effect(*args, **kwargs):
        command_args = args[0] if args else kwargs.get('args', [])
        env = kwargs.get('env', {})
        is_lean_libdir = command_args and command_args[0].endswith('lean') and command_args[1:] == ['--print-libdir']
        if is_lean_libdir:
            if lean_libdir_error: raise lean_libdir_error
            return mock_completed_process(returncode=0, stdout=lean_libdir)
        is_lake_build = command_args and len(command_args) >= 2 and command_args[0].endswith('lake') and command_args[1] == 'build'
        if is_lake_build:
            if lake_cache_path:
                assert 'LAKE_HOME' in env
                # Compare resolved paths for robustness
                assert pathlib.Path(env['LAKE_HOME']).resolve() == pathlib.Path(lake_cache_path).resolve()
            else:
                assert 'LAKE_HOME' not in env
            if lake_exception: raise lake_exception
            return mock_completed_process(returncode=lake_returncode, stdout=lake_stdout, stderr=lake_stderr)
        if command_args and command_args[0] in ['which', 'shutil.which']:
             return mock_completed_process(0, stdout='/fake/path/to/lean')
        print(f"WARNING: Unexpected subprocess call in mock: Args={args}, Kwargs={kwargs}")
        return mock_completed_process(returncode=1, stderr=f"Unexpected subprocess call: {command_args}")

    mock_run = mocker.patch('subprocess.run', side_effect=subprocess_side_effect)
    # Return the mock_makedirs to check calls if needed
    return mock_run, mock_makedirs


@patch.object(kb_storage_module, 'isinstance', lambda obj, cls: True)
class TestCheckAndCompileItem:

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    @patch('pathlib.Path.read_bytes')
    @patch('pathlib.Path.is_file') # Keep this for checking olean existence after build
    async def test_compile_success_cache_not_set(
        self, mock_is_file_olean, mock_read_bytes, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items[item_name])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_nocache.sqlite"
        temp_path_base = "/tmp/compile_nocache"
        lib_name = "TestLib"
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, _ = configure_mocks_for_compile(mocker, lake_cache_path=None, lake_returncode=0, lake_stdout="Build succeeded")
        # Mock olean check after successful build
        mock_is_file_olean.return_value = True
        mock_olean_content = b'\x01\x02\x03'
        mock_read_bytes.return_value = mock_olean_content
        mock_save.return_value = mock_item
        success, message = await check_and_compile_item(item_name, db_path=db_path, temp_lib_name=lib_name)
        assert success is True
        assert "Lake build successful" in message
        mock_run.assert_called()
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.PROVEN.name
        assert saved_item.lean_olean == mock_olean_content

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    @patch('pathlib.Path.read_bytes')
    @patch('pathlib.Path.is_file')
    async def test_compile_success_cache_set_exists(
        self, mock_is_file_olean, mock_read_bytes, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items[item_name])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_cache_exists.sqlite"
        temp_path_base = "/tmp/compile_cache_exists"
        lib_name = "TestLib"
        cache_path = "/persistent/lake/cache"
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, mock_makedirs = configure_mocks_for_compile(mocker, lake_cache_path=cache_path, cache_dir_exists=True, lake_returncode=0)
        mock_is_file_olean.return_value = True
        mock_olean_content = b'\x04\x05\x06'
        mock_read_bytes.return_value = mock_olean_content
        mock_save.return_value = mock_item
        success, message = await check_and_compile_item(item_name, db_path=db_path, temp_lib_name=lib_name)
        assert success is True
        mock_run.assert_called()
        mock_makedirs.assert_not_called() # Check that makedirs wasn't called
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.PROVEN.name
        assert saved_item.lean_olean == mock_olean_content

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    @patch('pathlib.Path.read_bytes')
    @patch('pathlib.Path.is_file')
    async def test_compile_success_cache_set_created(
        self, mock_is_file_olean, mock_read_bytes, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items[item_name])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_cache_create.sqlite"
        temp_path_base = "/tmp/compile_cache_create"
        lib_name = "TestLib"
        cache_path = "/new/persistent/lake/cache"
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, mock_makedirs = configure_mocks_for_compile(mocker, lake_cache_path=cache_path, cache_dir_exists=False, cache_creation_error=None, lake_returncode=0)
        mock_is_file_olean.return_value = True
        mock_olean_content = b'\x07\x08\x09'
        mock_read_bytes.return_value = mock_olean_content
        mock_save.return_value = mock_item
        success, message = await check_and_compile_item(item_name, db_path=db_path, temp_lib_name=lib_name)
        assert success is True
        mock_run.assert_called()
        # Check makedirs was called correctly
        mock_makedirs.assert_called_once_with(pathlib.Path(cache_path).resolve(), exist_ok=True)
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.PROVEN.name
        assert saved_item.lean_olean == mock_olean_content

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    async def test_compile_fail_cache_creation_error(
        self, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items[item_name])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_cache_createfail.sqlite"
        temp_path_base = "/tmp/compile_cache_createfail"
        lib_name = "TestLib"
        cache_path = "/unwritable/persistent/lake/cache"
        creation_error = OSError("Permission denied")
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, mock_makedirs = configure_mocks_for_compile(mocker, lake_cache_path=cache_path, cache_dir_exists=False, cache_creation_error=creation_error, lake_returncode=0)
        mocker.patch('pathlib.Path.is_file').return_value = True # Mock olean check
        mocker.patch('pathlib.Path.read_bytes').return_value = b'fake_olean' # Mock olean read
        mock_save.return_value = mock_item
        mock_log_error = mocker.patch.object(lean_interaction_logger, 'error')
        success, message = await check_and_compile_item(item_name, db_path=db_path, temp_lib_name=lib_name)
        assert success is True # Build succeeds even if cache creation fails
        mock_run.assert_called()
        mock_makedirs.assert_called_once_with(pathlib.Path(cache_path).resolve(), exist_ok=True)
        mock_log_error.assert_called_once_with(f"Failed to create persistent Lake cache directory '{pathlib.Path(cache_path).resolve()}': {creation_error}. Lake caching may not work as intended.")
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.PROVEN.name

    # --- Rest of the failure tests ---
    # Use .name for status assertions

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    async def test_check_compile_fail_compile_error(
        self, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.B"
        mock_item_B = copy.deepcopy(mock_kb_items["Lib.B"])
        mock_item_A = copy.deepcopy(mock_kb_items["Lib.A"])
        mock_item_B.status = ItemStatus.LEAN_VALIDATION_PENDING
        db_path = "/fake/db_compilefail.sqlite"
        temp_path_base = "/tmp/compile_fail"
        lib_name = "TestLib"
        error_stdout = "Compiling B..."
        error_stderr = f"./{lib_name}/{item_name.replace('.', '/')}.lean:1:10: error: unknown identifier 'oops'"
        full_error_output = f"--- STDOUT ---\n{error_stdout}\n--- STDERR ---\n{error_stderr}"
        mock_get_item.side_effect = lambda name, db_path=None: {"Lib.B": mock_item_B, "Lib.A": mock_item_A}.get(name)
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, _ = configure_mocks_for_compile(mocker, lake_cache_path=None, lake_returncode=1, lake_stdout=error_stdout, lake_stderr=error_stderr)
        mock_save.return_value = mock_item_B
        success, message = await check_and_compile_item(item_name, db_path=db_path, temp_lib_name=lib_name)
        assert success is False
        assert "Lean validation failed" in message
        mock_run.assert_called()
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.LEAN_VALIDATION_FAILED.name
        assert saved_item.lean_error_log == full_error_output.strip()

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    async def test_check_compile_fail_timeout(
        self, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items["Lib.A"])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_timeout.sqlite"
        timeout = 30
        temp_path_base = "/tmp/compile_timeout"
        lib_name = "TestLib"
        timeout_error = subprocess.TimeoutExpired(cmd=['lake', 'build', f"{lib_name}.{item_name}"], timeout=timeout)
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, _ = configure_mocks_for_compile(mocker, lake_cache_path=None, lake_exception=timeout_error)
        mock_save.return_value = mock_item
        success, message = await check_and_compile_item(item_name, db_path=db_path, timeout_seconds=timeout, temp_lib_name=lib_name)
        assert success is False
        assert f"Timeout after {timeout}s" in message
        mock_run.assert_called()
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.ERROR.name
        assert f"Timeout after {timeout}s" in saved_item.lean_error_log

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    async def test_check_compile_fail_lake_not_found(
        self, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items["Lib.A"])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_lakenotfound.sqlite"
        bad_lake_exec = "/usr/bin/nonexistent_lake"
        temp_path_base = "/tmp/compile_notfound"
        lib_name = "TestLib"
        file_not_found_error = FileNotFoundError(f"[Errno 2] No such file or directory: '{bad_lake_exec}'")
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run = mocker.patch('subprocess.run')
        def run_side_effect_lake_not_found(*args, **kwargs):
            command_args = args[0] if args else kwargs.get('args', [])
            if command_args and '--print-libdir' in command_args: return mock_completed_process(0, stdout="/fake/lib")
            elif command_args and command_args[0] == bad_lake_exec: raise file_not_found_error
            elif command_args and command_args[0] in ['which', 'shutil.which']: return mock_completed_process(0, stdout='/fake/path/to/lean')
            print(f"WARNING: Unexpected subprocess call in lake_not_found mock: {command_args}")
            return mock_completed_process(1, stderr="Unexpected call")
        mock_run.side_effect = run_side_effect_lake_not_found
        success, message = await check_and_compile_item(item_name, db_path=db_path, lake_executable_path=bad_lake_exec, temp_lib_name=lib_name)
        assert success is False
        assert "Lake executable not found" in message
        assert bad_lake_exec in message
        mock_run.assert_called()
        mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    async def test_check_compile_fail_item_not_found_db(self, mock_get_item, mock_save, mocker):
        item_name = "NonExistent.Item"
        db_path = "/fake/db_itemnotfound.sqlite"
        mock_get_item.return_value = None
        success, message = await check_and_compile_item(item_name, db_path=db_path)
        assert success is False
        assert f"Target item '{item_name}' not found" in message
        mock_get_item.assert_called_once_with(item_name, db_path=db_path)
        mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    async def test_check_compile_fail_dependency_fetch_error(
        self, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "ItemZ_MissingDep"
        db_path = "/fake/db_depfetchfail.sqlite"
        mock_item_Z = copy.deepcopy(mock_kb_items[item_name])
        fetch_error_msg = "Item 'MissingDep' not found"
        def get_item_side_effect(name, db_path=None):
            if name == item_name: return mock_item_Z
            elif name == "MissingDep": return None
            return None
        mock_get_item.side_effect = get_item_side_effect
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = "/tmp/compile_depfail"
        mock_save.return_value = mock_item_Z
        success, message = await check_and_compile_item(item_name, db_path=db_path)
        assert success is False
        assert "Dependency error" in message
        assert fetch_error_msg in message
        assert mock_get_item.call_count >= 2
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.ERROR.name
        assert f"Dependency fetch error: Item 'MissingDep' not found" in saved_item.lean_error_log

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    @patch('src.lean_automator.lean_interaction._create_temp_env_lake')
    async def test_check_compile_fail_env_creation_error(
        self, mock_create_env, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.B"
        mock_item_B = copy.deepcopy(mock_kb_items["Lib.B"])
        mock_item_A = copy.deepcopy(mock_kb_items["Lib.A"])
        db_path = "/fake/db_envcreatefail.sqlite"
        mock_item_B.status = ItemStatus.PENDING
        env_error = OSError("Permission denied creating fake dir")
        mock_get_item.side_effect = lambda name, db_path=None: {"Lib.B": mock_item_B, "Lib.A": mock_item_A}.get(name)
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = "/tmp/compile_envfail"
        mock_tempdir_instance.cleanup.return_value = None
        mock_create_env.side_effect = env_error
        mock_save.return_value = mock_item_B
        success, message = await check_and_compile_item(item_name, db_path=db_path)
        assert success is False
        assert "Environment creation error" in message
        assert str(env_error) in message
        assert mock_get_item.call_count >= 2
        mock_create_env.assert_called_once()
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.ERROR.name
        assert str(env_error) in saved_item.lean_error_log

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.rglob')
    async def test_check_compile_success_but_olean_not_found(
        self, mock_rglob, mock_is_file, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items["Lib.A"])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_oleannotfound.sqlite"
        temp_path_base = "/tmp/compile_oleanmissing"
        lib_name = "TestLib"
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, _ = configure_mocks_for_compile(mocker, lake_cache_path=None, lake_returncode=0, lake_stdout="Build ok")
        mock_is_file.return_value = False
        mock_rglob.return_value = [pathlib.Path(temp_path_base) / "lakefile.lean"]
        mock_save.return_value = mock_item
        success, message = await check_and_compile_item(item_name, db_path=db_path, temp_lib_name=lib_name)
        assert success is False
        assert "Lake build succeeded (exit 0)" in message
        assert ".olean file was not found" in message
        mock_run.assert_called()
        mock_is_file.assert_called()
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.ERROR.name
        assert ".olean file was not found" in saved_item.lean_error_log

    @pytest.mark.asyncio
    @patch('src.lean_automator.lean_interaction.save_kb_item', new_callable=AsyncMock)
    @patch('src.lean_automator.lean_interaction.get_kb_item_by_name')
    @patch('src.lean_automator.lean_interaction.tempfile.TemporaryDirectory')
    @patch('pathlib.Path.read_bytes')
    @patch('pathlib.Path.is_file')
    async def test_check_compile_success_olean_read_error(
        self, mock_is_file, mock_read_bytes, mock_tempdir, mock_get_item, mock_save, mocker, mock_kb_items):
        item_name = "Lib.A"
        mock_item = copy.deepcopy(mock_kb_items["Lib.A"])
        mock_item.status = ItemStatus.PENDING
        db_path = "/fake/db_oleanreadfail.sqlite"
        temp_path_base = "/tmp/compile_oleanreadfail"
        lib_name = "TestLib"
        read_error = OSError("Disk read error - permission denied")
        mock_get_item.return_value = mock_item
        mock_tempdir_instance = mock_tempdir.return_value
        mock_tempdir_instance.name = temp_path_base
        mock_run, _ = configure_mocks_for_compile(mocker, lake_cache_path=None, lake_returncode=0, lake_stdout="Build ok")
        mock_is_file.return_value = True
        mock_read_bytes.side_effect = read_error
        mock_save.return_value = mock_item
        success, message = await check_and_compile_item(item_name, db_path=db_path, temp_lib_name=lib_name)
        assert success is False
        assert "Error reading generated .olean" in message
        assert str(read_error) in message
        mock_run.assert_called()
        mock_is_file.assert_called()
        mock_read_bytes.assert_called_once()
        mock_save.assert_awaited_once()
        saved_item = mock_save.call_args[0][0]
        assert saved_item.status.name == ItemStatus.ERROR.name
        assert "Error reading generated .olean" in saved_item.lean_error_log
        assert str(read_error) in saved_item.lean_error_log