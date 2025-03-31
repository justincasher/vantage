# File: tests/integration/test_kb_storage_integration.py

import os
import sqlite3
import pytest
import json
import numpy as np # Added for embedding tests
import asyncio # Added for async calls
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import time # Import time for sleep

# Import the module to be tested, assuming 'src' is discoverable or using path adjustments
# If running pytest from the root directory, it should find lean_automator under src
from lean_automator import kb_storage
from lean_automator.kb_storage import (
    KBItem, ItemType, ItemStatus, LatexLink,
    DEFAULT_DB_PATH, # Import the default path for comparison if needed
    EMBEDDING_DTYPE # Import dtype for embedding tests
)

# --- Apply the integration mark to all tests in this module ---
pytestmark = pytest.mark.integration


# --- Test Configuration ---
# Use a dedicated file path for test database
TEST_DB_FILENAME = "test_integration_kb.sqlite"

# --- Pytest Fixture for Database Setup/Teardown ---

@pytest.fixture(scope="function") # Use "function" scope to get a fresh DB for each test
def test_db() -> str:
    """
    Pytest fixture to set up and tear down the test SQLite database.

    - Deletes any existing test database file before the test.
    - Initializes the database schema using kb_storage.initialize_database.
    - Yields the path to the test database file for the test function.
    - Deletes the test database file after the test completes.
    """
    db_path = TEST_DB_FILENAME
    # --- Setup ---
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError as e:
            pytest.fail(f"Failed to remove existing test database '{db_path}': {e}")

    try:
        # Use the updated initialize_database which handles new columns
        kb_storage.initialize_database(db_path=db_path)
    except Exception as e:
        pytest.fail(f"Failed to initialize test database '{db_path}': {e}")

    # --- Yield control to the test function ---
    yield db_path

    # --- Teardown ---
    # Use asyncio.run to potentially handle async cleanup if needed in future, though os.remove is sync
    # For now, sync removal is fine.
    async def _cleanup():
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except OSError as e:
            print(f"\nWarning: Could not clean up test database '{db_path}': {e}")
    asyncio.run(_cleanup())


# --- Test Functions ---

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_initialization(test_db):
    """Verify that the database file is created and schema initialized correctly."""
    assert os.path.exists(test_db), "Database file should exist after initialization"

    # Check if new columns exist after initialization
    with kb_storage.get_db_connection(test_db) as conn:
        cursor = conn.cursor()
        table_info = cursor.execute("PRAGMA table_info(kb_items);").fetchall()
        column_names = [col['name'] for col in table_info]
        assert 'latex_statement' in column_names # Changed from latex_exposition
        assert 'latex_proof' in column_names     # Check for latex_proof
        # assert 'lean_olean' in column_names # Removed lean_olean check
        assert 'plan_dependencies' in column_names
        assert 'failure_count' in column_names
        assert 'embedding_latex' in column_names # Check new embedding column
        assert 'embedding_nl' in column_names    # Check new embedding column
        assert 'latex_review_feedback' in column_names # Check new feedback column

    # Verify idempotency: initializing again should not cause errors
    try:
        kb_storage.initialize_database(db_path=test_db)
    except Exception as e:
        pytest.fail(f"Initializing database second time failed: {e}")

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_save_and_retrieve_new_item(test_db):
    """Test saving a new KBItem (with new fields) and retrieving it."""
    plan_deps = ["Prerequisite.Goal.A"]
    item = KBItem(
        unique_name="test.def.item_v2",
        item_type=ItemType.DEFINITION,
        latex_statement="LaTeX source for Item V2 definition.", # Changed from latex_exposition
        # latex_proof is None for DEFINITION
        lean_code="def ItemV2 := Nat",
        description_nl="Basic definition of ItemV2.",
        topic="Core.V2",
        # status defaults to PENDING
        plan_dependencies=plan_deps, # Add plan dependencies
        dependencies=[], # Lean dependencies (initially empty)
        latex_links=[LatexLink(citation_text="Ref B, Def 2")]
        # failure_count defaults to 0
    )

    # Save the item - use await and pass client=None to bypass embedding generation
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)

    # --- Assertions after save ---
    assert saved_item.id is not None, "ID should be assigned after saving"
    original_id = saved_item.id

    # --- Test retrieval by ID ---
    # Retrieval functions are synchronous
    retrieved_by_id = kb_storage.get_kb_item_by_id(original_id, db_path=test_db)
    assert retrieved_by_id is not None, "Should retrieve item by ID"
    assert retrieved_by_id.id == original_id
    assert retrieved_by_id.unique_name == "test.def.item_v2"
    assert retrieved_by_id.item_type == ItemType.DEFINITION
    assert retrieved_by_id.latex_statement == "LaTeX source for Item V2 definition." # Changed from latex_exposition
    assert retrieved_by_id.latex_proof is None # Should be None for definition
    assert retrieved_by_id.lean_code == "def ItemV2 := Nat"
    assert retrieved_by_id.description_nl == "Basic definition of ItemV2."
    assert retrieved_by_id.topic == "Core.V2"
    assert retrieved_by_id.status == ItemStatus.PENDING # Check default status
    assert retrieved_by_id.plan_dependencies == plan_deps # Check plan dependencies
    assert retrieved_by_id.failure_count == 0 # Check default failure count
    assert retrieved_by_id.dependencies == []
    assert len(retrieved_by_id.latex_links) == 1
    assert retrieved_by_id.latex_links[0].citation_text == "Ref B, Def 2"
    # assert retrieved_by_id.lean_olean is None # Check default # Removed lean_olean check
    assert retrieved_by_id.embedding_latex is None # Check default
    assert retrieved_by_id.embedding_nl is None # Check default
    assert retrieved_by_id.latex_review_feedback is None # Check default
    assert retrieved_by_id.created_at is not None
    assert retrieved_by_id.last_modified_at is not None
    assert retrieved_by_id.created_at.tzinfo == timezone.utc
    assert retrieved_by_id.last_modified_at.tzinfo == timezone.utc

    # --- Test retrieval by unique_name ---
    retrieved_by_name = kb_storage.get_kb_item_by_name("test.def.item_v2", db_path=test_db)
    assert retrieved_by_name is not None, "Should retrieve item by unique_name"
    assert retrieved_by_name.id == original_id
    assert retrieved_by_name.plan_dependencies == plan_deps
    assert retrieved_by_name.failure_count == 0

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_update_item(test_db):
    """Test updating an existing KBItem's attributes, including new fields."""
    # First, save an initial item
    initial_plan_deps = ["dep.goal.1"]
    item = KBItem(
        unique_name="test.thm.update_v2",
        item_type=ItemType.THEOREM,
        latex_statement="Initial LaTeX Statement V2.", # Changed from latex_exposition
        latex_proof="Initial LaTeX Proof V2.",         # Add proof for theorem
        lean_code="theorem update_v2 : Nat := 0", # Simplified Lean code
        topic="Test.UpdateV2",
        status=ItemStatus.PENDING, # Start as PENDING
        plan_dependencies=initial_plan_deps,
        failure_count=0
    )
    # Use await and client=None
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    original_id = saved_item.id
    original_mod_time = saved_item.last_modified_at

    # Wait slightly to ensure timestamp changes
    await asyncio.sleep(0.01)

    # Modify the retrieved item
    retrieved_item = kb_storage.get_kb_item_by_id(original_id, db_path=test_db)
    assert retrieved_item is not None
    retrieved_item.status = ItemStatus.PENDING_LATEX_REVIEW # Update status (was LATEX_GENERATED)
    retrieved_item.description_nl = "Updated description V2."
    retrieved_item.add_dependency("lean.dep.x") # Add Lean dependency
    retrieved_item.add_plan_dependency("dep.goal.2") # Add plan dependency
    retrieved_item.increment_failure_count() # Increment failure count
    retrieved_item.latex_statement = "Revised LaTeX Statement V2." # Changed from latex_exposition
    retrieved_item.latex_proof = "Revised LaTeX Proof V2." # Update proof
    # retrieved_item.lean_olean = b'new_olean_data' # Add olean data # Removed lean_olean assignment
    retrieved_item.latex_review_feedback = "Typo in definition X." # Add feedback

    # Save the modified item (should trigger an UPDATE) - use await and client=None
    updated_item = await kb_storage.save_kb_item(retrieved_item, client=None, db_path=test_db)

    # --- Assertions after update ---
    assert updated_item.id == original_id
    assert updated_item.status == ItemStatus.PENDING_LATEX_REVIEW
    assert updated_item.description_nl == "Updated description V2."
    assert updated_item.dependencies == ["lean.dep.x"]
    assert updated_item.plan_dependencies == ["dep.goal.1", "dep.goal.2"] # Check combined list
    assert updated_item.failure_count == 1 # Check incremented value
    assert updated_item.latex_statement == "Revised LaTeX Statement V2." # Changed from latex_exposition
    assert updated_item.latex_proof == "Revised LaTeX Proof V2." # Check proof update
    # assert updated_item.lean_olean == b'new_olean_data' # Check olean data # Removed lean_olean check
    assert updated_item.latex_review_feedback == "Typo in definition X." # Check feedback
    assert updated_item.last_modified_at > original_mod_time
    assert updated_item.created_at == retrieved_item.created_at

    # --- Verify persistence by retrieving again ---
    retrieved_again = kb_storage.get_kb_item_by_id(original_id, db_path=test_db)
    assert retrieved_again is not None
    assert retrieved_again.status == ItemStatus.PENDING_LATEX_REVIEW
    assert retrieved_again.latex_statement == "Revised LaTeX Statement V2." # Changed from latex_exposition
    assert retrieved_again.latex_proof == "Revised LaTeX Proof V2."
    assert retrieved_again.plan_dependencies == ["dep.goal.1", "dep.goal.2"]
    assert retrieved_again.failure_count == 1
    # assert retrieved_again.lean_olean == b'new_olean_data' # Removed lean_olean check
    assert retrieved_again.latex_review_feedback == "Typo in definition X."

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_unique_name_conflict_upsert(test_db):
    """Test UPSERT updates all fields including new ones."""
    # Save initial item (Theorem requires proof)
    item1 = KBItem(
        unique_name="test.upsert.conflict.v2",
        item_type=ItemType.THEOREM, # Needs proof
        latex_statement="V1 LaTeX Statement", # Changed from latex_exposition
        latex_proof="V1 Proof",
        lean_code="Version 1",
        status=ItemStatus.PENDING,
        plan_dependencies=["depA"],
        failure_count=0,
        # lean_olean=b'old_olean', # Add initial olean # Removed lean_olean
        latex_review_feedback="Initial Feedback"
    )
    # Use await and client=None
    saved1 = await kb_storage.save_kb_item(item1, client=None, db_path=test_db)
    assert saved1.id is not None
    original_mod_time = saved1.last_modified_at
    await asyncio.sleep(0.01)

    # Save second item with SAME unique_name but different content
    item2 = KBItem(
        unique_name="test.upsert.conflict.v2",
        item_type=ItemType.THEOREM, # Keep type consistent for proof field
        latex_statement="V2 LaTeX Statement Updated", # Changed from latex_exposition
        latex_proof="V2 Proof Updated",
        lean_code="Version 2",
        status=ItemStatus.PENDING_LATEX_REVIEW, # Was LATEX_GENERATED
        plan_dependencies=["depB", "depC"], # Different plan deps
        failure_count=1, # Different failure count
        # lean_olean=b'new_olean', # Different olean # Removed lean_olean
        latex_review_feedback="Updated Feedback" # Different feedback
    )
    # Use await and client=None
    saved2 = await kb_storage.save_kb_item(item2, client=None, db_path=test_db)

    # --- Assertions ---
    assert saved2.id is not None
    assert saved2.id == saved1.id # Check ID reused
    assert saved2.lean_code == "Version 2"
    assert saved2.latex_statement == "V2 LaTeX Statement Updated" # Changed from latex_exposition
    assert saved2.latex_proof == "V2 Proof Updated"
    assert saved2.status == ItemStatus.PENDING_LATEX_REVIEW
    assert saved2.plan_dependencies == ["depB", "depC"] # Check plan deps updated
    assert saved2.failure_count == 1 # Check failure count updated
    # assert saved2.lean_olean == b'new_olean' # Check olean updated # Removed lean_olean check
    assert saved2.latex_review_feedback == "Updated Feedback" # Check feedback updated
    assert saved2.last_modified_at > original_mod_time

    # Verify by retrieving
    retrieved = kb_storage.get_kb_item_by_name("test.upsert.conflict.v2", db_path=test_db)
    assert retrieved is not None
    assert retrieved.id == saved1.id
    assert retrieved.lean_code == "Version 2"
    assert retrieved.latex_statement == "V2 LaTeX Statement Updated" # Changed from latex_exposition
    assert retrieved.latex_proof == "V2 Proof Updated"
    assert retrieved.status == ItemStatus.PENDING_LATEX_REVIEW
    assert retrieved.plan_dependencies == ["depB", "depC"]
    assert retrieved.failure_count == 1
    # assert retrieved.lean_olean == b'new_olean' # Removed lean_olean check
    assert retrieved.latex_review_feedback == "Updated Feedback"

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_get_items_by_status(test_db):
    """Test retrieving items filtered by status, using updated statuses."""
    # Save items with different statuses - use await and client=None
    await kb_storage.save_kb_item(KBItem(unique_name="t_status.pending1", status=ItemStatus.PENDING), client=None, db_path=test_db)
    await kb_storage.save_kb_item(KBItem(unique_name="t_status.pending2", status=ItemStatus.PENDING), client=None, db_path=test_db)
    await kb_storage.save_kb_item(KBItem(unique_name="t_status.proven", status=ItemStatus.PROVEN), client=None, db_path=test_db)
    await kb_storage.save_kb_item(KBItem(unique_name="t_status.latex_review", status=ItemStatus.PENDING_LATEX_REVIEW), client=None, db_path=test_db) # Was LATEX_GENERATED
    await kb_storage.save_kb_item(KBItem(unique_name="t_status.latex_accepted", status=ItemStatus.LATEX_ACCEPTED), client=None, db_path=test_db) # Was LATEX_REVIEW_PASSED
    await kb_storage.save_kb_item(KBItem(unique_name="t_status.lean_fail", status=ItemStatus.LEAN_VALIDATION_FAILED), client=None, db_path=test_db)
    await kb_storage.save_kb_item(KBItem(unique_name="t_status.lean_gen", status=ItemStatus.LEAN_GENERATION_IN_PROGRESS), client=None, db_path=test_db) # Add a new status

    # Retrieval function is synchronous
    # --- Test retrieval with PENDING status ---
    pending_items = list(kb_storage.get_items_by_status(ItemStatus.PENDING, db_path=test_db))
    pending_names = {item.unique_name for item in pending_items}
    assert len(pending_items) == 2, "Should find exactly 2 PENDING items"
    assert "t_status.pending1" in pending_names
    assert "t_status.pending2" in pending_names

    # --- Test retrieval with other statuses (check counts only) ---
    proven_items = list(kb_storage.get_items_by_status(ItemStatus.PROVEN, db_path=test_db))
    assert len(proven_items) == 1

    latex_review_items = list(kb_storage.get_items_by_status(ItemStatus.PENDING_LATEX_REVIEW, db_path=test_db))
    assert len(latex_review_items) == 1

    latex_accepted_items = list(kb_storage.get_items_by_status(ItemStatus.LATEX_ACCEPTED, db_path=test_db))
    assert len(latex_accepted_items) == 1

    lean_fail_items = list(kb_storage.get_items_by_status(ItemStatus.LEAN_VALIDATION_FAILED, db_path=test_db))
    assert len(lean_fail_items) == 1

    lean_gen_items = list(kb_storage.get_items_by_status(ItemStatus.LEAN_GENERATION_IN_PROGRESS, db_path=test_db))
    assert len(lean_gen_items) == 1

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_get_items_by_topic(test_db):
    """Test retrieving items filtered by topic prefix."""
    # Use await and client=None
    await kb_storage.save_kb_item(KBItem(unique_name="t_topic.core.types", topic="Core.Types"), client=None, db_path=test_db)
    await kb_storage.save_kb_item(KBItem(unique_name="t_topic.core.nat", topic="Core.Nat"), client=None, db_path=test_db)
    await kb_storage.save_kb_item(KBItem(unique_name="t_topic.algebra.groups", topic="Algebra.Groups"), client=None, db_path=test_db)
    await kb_storage.save_kb_item(KBItem(unique_name="t_topic.core", topic="Core"), client=None, db_path=test_db)

    # Retrieval is synchronous
    core_items = list(kb_storage.get_items_by_topic("Core", db_path=test_db))
    assert len(core_items) == 3

    core_nat_items = list(kb_storage.get_items_by_topic("Core.Nat", db_path=test_db))
    assert len(core_nat_items) == 1


def test_retrieve_non_existent(test_db):
    """Test retrieving non-existent items returns None."""
    # No async calls here, no changes needed
    retrieved_id = kb_storage.get_kb_item_by_id(99999, db_path=test_db)
    assert retrieved_id is None

    retrieved_name = kb_storage.get_kb_item_by_name("test.non_existent.name", db_path=test_db)
    assert retrieved_name is None

# Mark tests that call async functions
# @pytest.mark.asyncio
# async def test_save_and_retrieve_olean(test_db): # Removed entire test
#    """Test saving and retrieving binary lean_olean data."""
#    ...

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_save_and_retrieve_embeddings(test_db):
    """Test saving and retrieving binary embedding data (when manually set)."""
    # Create some dummy embedding data
    embedding_data_latex = np.array([0.1, -0.2, 0.3], dtype=EMBEDDING_DTYPE).tobytes()
    embedding_data_nl = np.array([0.4, 0.5, -0.6], dtype=EMBEDDING_DTYPE).tobytes()

    item = KBItem(
        unique_name="test.with.embeddings",
        item_type=ItemType.EXAMPLE, # Example requires proof/verification
        latex_statement="Example statement",
        latex_proof="Example proof",
        lean_code="example : 1 = 1 := rfl",
        description_nl="An example item",
        # Explicitly set embedding data before saving
        embedding_latex=embedding_data_latex,
        embedding_nl=embedding_data_nl,
        status=ItemStatus.PROVEN
    )
    # Use await and client=None - this should NOT regenerate embeddings
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    assert saved_item.id is not None
    # Check that save_kb_item didn't overwrite the manually set embeddings
    assert saved_item.embedding_latex == embedding_data_latex
    assert saved_item.embedding_nl == embedding_data_nl
    assert saved_item.latex_statement == "Example statement"
    assert saved_item.latex_proof == "Example proof"
    assert saved_item.description_nl == "An example item"

    # Retrieval is synchronous
    retrieved_item = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved_item is not None
    # Verify the retrieved blobs match the original blobs
    assert retrieved_item.embedding_latex == embedding_data_latex
    assert retrieved_item.embedding_nl == embedding_data_nl
    assert retrieved_item.latex_statement == "Example statement"
    assert retrieved_item.latex_proof == "Example proof"
    assert retrieved_item.description_nl == "An example item"

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_complex_dependencies(test_db):
    """Test saving and retrieving item with multiple complex Lean dependencies."""
    deps = ["Core.Init.Default", "Algebra.Group.Defs", "Topology.Basic", "MyProject.Module.Sub"]
    item = KBItem(
        unique_name="test.complex.deps",
        item_type=ItemType.STRUCTURE, # Does not require proof
        dependencies=deps, # Testing the Lean dependencies field
        latex_statement="Structure statement", # Add required field
        status=ItemStatus.PROVEN # Using alias STRUCTURE_ADDED points to PROVEN
    )
    # Use await and client=None
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    assert saved_item.id is not None

    # Retrieval is synchronous
    retrieved_item = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved_item is not None
    assert retrieved_item.dependencies == deps
    assert retrieved_item.latex_proof is None # Check proof is None

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_complex_latex_links(test_db):
    """Test saving and retrieving item with multiple detailed LatexLinks."""
    links = [
        LatexLink(citation_text="S1", link_type="definition", source_identifier="DOI:X"),
        LatexLink(citation_text="S2", link_type="proof", source_identifier="URL:Y", verified_by_human=True),
        LatexLink(citation_text="S3")
    ]
    item = KBItem(
        unique_name="test.complex.links",
        item_type=ItemType.LEMMA, # Requires proof
        latex_statement="Lemma statement",
        latex_proof="Lemma proof",
        latex_links=links,
        status=ItemStatus.PROVEN
    )
    # Use await and client=None
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    assert saved_item.id is not None

    # Retrieval is synchronous
    retrieved_item = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved_item is not None
    assert len(retrieved_item.latex_links) == len(links)
    assert retrieved_item.latex_proof == "Lemma proof" # Check proof presence

    # Compare dictionary representations for equality check
    retrieved_links_dicts = [asdict(link) for link in retrieved_item.latex_links]
    original_links_dicts = [asdict(link) for link in links]
    assert retrieved_links_dicts == original_links_dicts


# --- New Tests for Plan Dependencies and Failure Count ---

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_add_plan_dependency(test_db):
    """Test the add_plan_dependency method and persistence."""
    item = KBItem(unique_name="test.plan.dep.target", item_type=ItemType.THEOREM) # Needs statement/proof
    # Use await and client=None
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    original_mod_time = saved_item.last_modified_at
    await asyncio.sleep(0.01)

    # Add first dependency
    saved_item.add_plan_dependency("plan.dep.1")
    # Use await and client=None
    updated_item1 = await kb_storage.save_kb_item(saved_item, client=None, db_path=test_db)
    assert updated_item1.plan_dependencies == ["plan.dep.1"]
    assert updated_item1.last_modified_at > original_mod_time

    # Add same dependency again
    mod_time_after_add1 = updated_item1.last_modified_at
    updated_item1.add_plan_dependency("plan.dep.1") # Should have no effect on list
    # Use await and client=None
    updated_item2 = await kb_storage.save_kb_item(updated_item1, client=None, db_path=test_db)
    assert updated_item2.plan_dependencies == ["plan.dep.1"]
    # Timestamp might update slightly due to resave, or might not if no change detected
    # Let's check it's at least the same or newer
    assert updated_item2.last_modified_at >= mod_time_after_add1

    # Add second dependency
    await asyncio.sleep(0.01)
    updated_item2.add_plan_dependency("plan.dep.2")
    # Use await and client=None
    updated_item3 = await kb_storage.save_kb_item(updated_item2, client=None, db_path=test_db)
    assert updated_item3.plan_dependencies == ["plan.dep.1", "plan.dep.2"]
    # Check time updated compared to *after* the no-op save
    assert updated_item3.last_modified_at > mod_time_after_add1 # updated_item2 timestamp might be same as updated_item1

    # Retrieve to confirm persistence
    retrieved = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved is not None
    assert retrieved.plan_dependencies == ["plan.dep.1", "plan.dep.2"]

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_increment_failure_count(test_db):
    """Test the increment_failure_count method and persistence."""
    item = KBItem(unique_name="test.failure.count", item_type=ItemType.LEMMA) # Needs statement/proof
    # Use await and client=None
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    assert saved_item.failure_count == 0 # Check initial default
    original_mod_time = saved_item.last_modified_at
    await asyncio.sleep(0.01)

    # Increment once
    saved_item.increment_failure_count()
    # Use await and client=None
    updated_item1 = await kb_storage.save_kb_item(saved_item, client=None, db_path=test_db)
    assert updated_item1.failure_count == 1
    assert updated_item1.last_modified_at > original_mod_time
    mod_time_after_inc1 = updated_item1.last_modified_at
    await asyncio.sleep(0.01)

    # Increment again
    updated_item1.increment_failure_count()
    # Use await and client=None
    updated_item2 = await kb_storage.save_kb_item(updated_item1, client=None, db_path=test_db)
    assert updated_item2.failure_count == 2
    assert updated_item2.last_modified_at > mod_time_after_inc1

    # Retrieve to confirm persistence
    retrieved = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved is not None
    assert retrieved.failure_count == 2

# Mark tests that call async functions
@pytest.mark.asyncio
async def test_item_type_requires_proof_handling(test_db):
    """Test that latex_proof is correctly handled based on item_type."""
    # 1. Item type that requires proof (Theorem)
    item_thm = KBItem(
        unique_name="test.proof.required",
        item_type=ItemType.THEOREM,
        latex_statement="Thm statement",
        latex_proof="Thm proof content" # Provide proof
    )
    saved_thm = await kb_storage.save_kb_item(item_thm, client=None, db_path=test_db)
    retrieved_thm = kb_storage.get_kb_item_by_id(saved_thm.id, db_path=test_db)
    assert retrieved_thm.item_type.requires_proof() is True
    assert retrieved_thm.latex_proof == "Thm proof content"

    # 2. Item type that does NOT require proof (Definition)
    item_def = KBItem(
        unique_name="test.proof.not_required",
        item_type=ItemType.DEFINITION,
        latex_statement="Def statement",
        latex_proof="Should be ignored" # Provide proof, but it should be nulled
    )
    # __post_init__ or save_kb_item should nullify latex_proof
    assert item_def.latex_proof is None # Check after __post_init__
    saved_def = await kb_storage.save_kb_item(item_def, client=None, db_path=test_db)
    retrieved_def = kb_storage.get_kb_item_by_id(saved_def.id, db_path=test_db)
    assert retrieved_def.item_type.requires_proof() is False
    assert retrieved_def.latex_proof is None # Verify it's None in DB/retrieval

    # 3. Update an item to a type that doesn't require proof
    retrieved_thm.item_type = ItemType.REMARK # Remark does not require proof
    retrieved_thm.latex_proof = "This should also be ignored now"
    # save_kb_item should nullify the proof during save
    updated_thm = await kb_storage.save_kb_item(retrieved_thm, client=None, db_path=test_db)
    retrieved_remark = kb_storage.get_kb_item_by_id(updated_thm.id, db_path=test_db)
    assert retrieved_remark.item_type == ItemType.REMARK
    assert retrieved_remark.item_type.requires_proof() is False
    assert retrieved_remark.latex_proof is None # Verify proof is gone