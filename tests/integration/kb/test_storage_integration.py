# File: tests/integration/kb/test_storage_integration.py

"""Integration tests for the Knowledge Base storage module (storage).

Verifies database interactions including initialization, saving new items,
updating existing items (UPSERT), retrieving items by various criteria,
and handling specific fields like embeddings, dependencies, links, and
status updates using a temporary SQLite database.
"""

import os
import sqlite3
import pytest
import json
import numpy as np  # Added for embedding tests
import asyncio  # Added for async calls
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import time  # Import time for sleep
import logging  # Import logging

# Import the module to be tested, assuming 'src' is discoverable or using path adjustments
# If running pytest from the root directory, it should find lean_automator under src
from lean_automator.kb import storage as kb_storage
from lean_automator.kb.storage import (
    KBItem,
    ItemType,
    ItemStatus,
    LatexLink,
    DEFAULT_DB_PATH,  # Import the default path for comparison if needed
    EMBEDDING_DTYPE,  # Import dtype for embedding tests
)

# --- Apply the integration mark to all tests in this module ---
pytestmark = pytest.mark.integration

# --- Logging ---
# Configure logging specifically for tests if needed, or rely on global config
logger = logging.getLogger(__name__)

# --- Test Configuration ---
# Use a dedicated file path for test database
TEST_DB_FILENAME = "test_integration_kb.sqlite"

# --- Pytest Fixture for Database Setup/Teardown ---


@pytest.fixture(
    scope="function"
)  # Use "function" scope to get a fresh DB for each test
def test_db() -> str:
    """Sets up and tears down the test SQLite database for each test function.

    Ensures a clean database state for each test by:
    1. Deleting any pre-existing test database file (`test_integration_kb.sqlite`).
    2. Calling `kb_storage.initialize_database` to create the schema.
    3. Yielding the path to the database file for use in the test.
    4. Deleting the database file after the test function completes.

    Yields:
        str: The file path to the initialized, empty test database.
    """
    db_path = TEST_DB_FILENAME
    logger.debug(f"Setting up test database: {db_path}")
    # --- Setup ---
    if os.path.exists(db_path):
        try:
            logger.debug(f"Removing existing test database: {db_path}")
            os.remove(db_path)
        except OSError as e:
            pytest.fail(f"Failed to remove existing test database '{db_path}': {e}")

    try:
        logger.debug(f"Initializing test database schema: {db_path}")
        # Use the updated initialize_database which handles new columns
        kb_storage.initialize_database(db_path=db_path)
    except Exception as e:
        pytest.fail(f"Failed to initialize test database '{db_path}': {e}")

    # --- Yield control to the test function ---
    yield db_path

    # --- Teardown ---
    # Use asyncio.run to potentially handle async cleanup if needed in future, though os.remove is sync
    # For now, sync removal is fine, but wrap in async context for consistency if other async cleanup needed
    async def _cleanup():
        try:
            if os.path.exists(db_path):
                logger.debug(f"Cleaning up test database: {db_path}")
                os.remove(db_path)
        except OSError as e:
            # Log warning but don't fail test run if cleanup fails
            logger.warning(f"Could not clean up test database '{db_path}': {e}")

    # Run the synchronous os.remove within an async context if needed, or just call directly
    # If no other async teardown is required, direct sync call is simpler:
    try:
        if os.path.exists(db_path):
            logger.debug(f"Cleaning up test database: {db_path}")
            os.remove(db_path)
    except OSError as e:
        logger.warning(f"Could not clean up test database '{db_path}': {e}")
    # asyncio.run(_cleanup()) # Keep if planning other async teardown


# --- Test Functions ---


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_initialization(test_db):
    """Verify database creation and correct schema initialization."""
    assert os.path.exists(test_db), "Database file should be created by fixture"

    # Check if expected columns exist
    with kb_storage.get_db_connection(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(kb_items);")
        columns = {row["name"] for row in cursor.fetchall()}

    expected_columns = {
        "id",
        "unique_name",
        "item_type",
        "description_nl",
        "latex_statement",
        "latex_proof",
        "lean_code",
        "embedding_nl",
        "embedding_latex",
        "topic",
        "plan_dependencies",
        "dependencies",
        "latex_links",
        "status",
        "failure_count",
        "latex_review_feedback",
        "generation_prompt",
        "raw_ai_response",
        "lean_error_log",
        "created_at",
        "last_modified_at",
    }
    assert expected_columns.issubset(columns), (
        f"Missing columns: {expected_columns - columns}"
    )

    # Verify idempotency: initializing again should not raise errors
    try:
        logger.debug("Testing database initialization idempotency...")
        kb_storage.initialize_database(db_path=test_db)
        logger.debug("Second initialization successful.")
    except Exception as e:
        pytest.fail(f"Initializing database second time failed unexpectedly: {e}")


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_save_and_retrieve_new_item(test_db):
    """Test saving a completely new KBItem and retrieving it accurately."""
    plan_deps = ["Prerequisite.Goal.A"]
    links = [LatexLink(citation_text="Ref B, Def 2")]
    item_time = datetime.now(timezone.utc) - timedelta(
        minutes=1
    )  # Ensure time is distinct

    item = KBItem(
        unique_name="test.def.item_v2",
        item_type=ItemType.DEFINITION,
        latex_statement="LaTeX source for Item V2 definition.",
        # latex_proof is None for DEFINITION type
        lean_code="def ItemV2 := Nat",
        description_nl="Basic definition of ItemV2.",
        topic="Core.V2",
        status=ItemStatus.PENDING,  # Explicitly set default status
        plan_dependencies=plan_deps,
        dependencies=[],
        latex_links=links,
        failure_count=0,  # Explicitly set default
        created_at=item_time,  # Control creation time for comparison
        last_modified_at=item_time,  # Control modification time for comparison
    )

    # Save item (client=None prevents embedding generation attempt)
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)

    # Assertions on saved item
    assert saved_item.id is not None, "ID should be assigned by DB"
    assert saved_item.created_at == item_time  # Should persist explicitly set time
    # last_modified_at might be updated by save_kb_item, check it's >= initial
    assert saved_item.last_modified_at >= item_time

    # Retrieve and compare all fields
    retrieved = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved is not None, "Failed to retrieve item by ID"

    # Use asdict for easier comparison, excluding potentially db-modified fields
    saved_dict = asdict(saved_item)
    retrieved_dict = asdict(retrieved)

    # Compare relevant fields explicitly or via dict comparison after removing mutable ones
    assert retrieved_dict["unique_name"] == saved_dict["unique_name"]
    assert (
        retrieved_dict["item_type"] == saved_dict["item_type"]
    )  # Enums compared directly
    assert retrieved_dict["latex_statement"] == saved_dict["latex_statement"]
    assert retrieved_dict["latex_proof"] is None  # Check None for DEFINITION
    assert retrieved_dict["lean_code"] == saved_dict["lean_code"]
    assert retrieved_dict["description_nl"] == saved_dict["description_nl"]
    assert retrieved_dict["topic"] == saved_dict["topic"]
    assert retrieved_dict["status"] == saved_dict["status"]  # Enums compared directly
    assert retrieved_dict["plan_dependencies"] == saved_dict["plan_dependencies"]
    assert retrieved_dict["dependencies"] == saved_dict["dependencies"]
    assert (
        retrieved_dict["latex_links"] == saved_dict["latex_links"]
    )  # Compare list of dataclasses
    assert retrieved_dict["failure_count"] == saved_dict["failure_count"]
    assert retrieved_dict["embedding_nl"] is None
    assert retrieved_dict["embedding_latex"] is None
    assert retrieved_dict["latex_review_feedback"] is None
    # Timestamps might differ slightly due to DB precision or save_kb_item updates
    assert abs(retrieved_dict["created_at"] - saved_dict["created_at"]) < timedelta(
        seconds=1
    )
    assert abs(
        retrieved_dict["last_modified_at"] - saved_dict["last_modified_at"]
    ) < timedelta(seconds=1)

    # Test retrieval by name
    retrieved_by_name = kb_storage.get_kb_item_by_name(
        "test.def.item_v2", db_path=test_db
    )
    assert retrieved_by_name is not None, "Failed to retrieve item by name"
    assert retrieved_by_name.id == saved_item.id


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_update_item(test_db):
    """Test updating various fields of an existing KBItem."""
    # Save initial item
    item = KBItem(unique_name="test.thm.update_v2", item_type=ItemType.THEOREM)
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    original_id = saved_item.id
    original_mod_time = saved_item.last_modified_at
    await asyncio.sleep(0.01)  # Ensure time progresses

    # Modify retrieved item
    saved_item.status = ItemStatus.PENDING_LATEX_REVIEW
    saved_item.description_nl = "Updated description V2."
    saved_item.add_dependency("lean.dep.x")
    saved_item.add_plan_dependency("dep.goal.2")
    saved_item.increment_failure_count()
    saved_item.latex_statement = "Revised LaTeX Statement V2."
    saved_item.latex_proof = "Revised LaTeX Proof V2."
    saved_item.latex_review_feedback = "Typo in definition X."

    # Save changes (UPSERT should trigger UPDATE)
    updated_item = await kb_storage.save_kb_item(
        saved_item, client=None, db_path=test_db
    )

    # Assertions on updated item
    assert updated_item.id == original_id
    assert updated_item.status == ItemStatus.PENDING_LATEX_REVIEW
    assert updated_item.description_nl == "Updated description V2."
    assert updated_item.dependencies == ["lean.dep.x"]
    # Note: KBItem methods append, plan_dependencies depends on initial state
    # Need to re-fetch initial state if testing specific append logic vs overwrite.
    # Assuming save_kb_item overwrites lists based on current object state:
    assert updated_item.plan_dependencies == [
        "dep.goal.2"
    ]  # Check if save overwrites or appends based on implementation detail
    # If save_kb_item *appends* based on instance methods, the initial [] would become ["dep.goal.2"]
    # If save_kb_item *overwrites* based on item.to_dict_for_db, it will be ["dep.goal.2"]
    # Let's assume overwrite for now.

    assert updated_item.failure_count == 1
    assert updated_item.latex_statement == "Revised LaTeX Statement V2."
    assert updated_item.latex_proof == "Revised LaTeX Proof V2."
    assert updated_item.latex_review_feedback == "Typo in definition X."
    assert updated_item.last_modified_at > original_mod_time

    # Verify persistence by retrieving again
    retrieved_again = kb_storage.get_kb_item_by_id(original_id, db_path=test_db)
    assert retrieved_again is not None
    assert retrieved_again.status == ItemStatus.PENDING_LATEX_REVIEW
    assert retrieved_again.failure_count == 1
    assert retrieved_again.plan_dependencies == [
        "dep.goal.2"
    ]  # Verify overwrite assumption
    assert retrieved_again.latex_review_feedback == "Typo in definition X."


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_unique_name_conflict_upsert(test_db):
    """Verify UPSERT behavior: saving with existing unique_name updates fields."""
    # Save initial item
    item1 = KBItem(
        unique_name="test.upsert.conflict.v2",
        item_type=ItemType.THEOREM,
        lean_code="V1",
        failure_count=0,
    )
    saved1 = await kb_storage.save_kb_item(item1, client=None, db_path=test_db)
    await asyncio.sleep(0.01)

    # Save second item with SAME unique_name but different content
    item2 = KBItem(
        unique_name="test.upsert.conflict.v2",
        item_type=ItemType.THEOREM,
        lean_code="V2",
        status=ItemStatus.PROVEN,
        failure_count=5,
    )
    saved2 = await kb_storage.save_kb_item(item2, client=None, db_path=test_db)

    # Assertions: Check ID is reused and fields are updated
    assert saved2.id == saved1.id, "ID should be reused on UPSERT conflict"
    assert saved2.lean_code == "V2", "lean_code should be updated"
    assert saved2.status == ItemStatus.PROVEN, "status should be updated"
    assert saved2.failure_count == 5, "failure_count should be updated"
    assert saved2.last_modified_at > saved1.last_modified_at, (
        "last_modified_at should update"
    )

    # Verify by retrieving
    retrieved = kb_storage.get_kb_item_by_name(
        "test.upsert.conflict.v2", db_path=test_db
    )
    assert retrieved is not None
    assert retrieved.id == saved1.id
    assert retrieved.lean_code == "V2"
    assert retrieved.status == ItemStatus.PROVEN
    assert retrieved.failure_count == 5


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_get_items_by_status(test_db):
    """Verify retrieving items filtered by their status."""
    # Save items with various statuses
    statuses_to_test = [
        ItemStatus.PENDING,
        ItemStatus.PROVEN,
        ItemStatus.LEAN_VALIDATION_FAILED,
        ItemStatus.LATEX_ACCEPTED,
    ]
    for i, status in enumerate(statuses_to_test):
        await kb_storage.save_kb_item(
            KBItem(unique_name=f"t_status.{status.name}_{i}", status=status),
            client=None,
            db_path=test_db,
        )
    # Add a duplicate status
    await kb_storage.save_kb_item(
        KBItem(unique_name="t_status.PENDING_1", status=ItemStatus.PENDING),
        client=None,
        db_path=test_db,
    )

    # Test retrieval for each status
    pending_items = list(
        kb_storage.get_items_by_status(ItemStatus.PENDING, db_path=test_db)
    )
    assert len(pending_items) == 2, "Expected 2 PENDING items"
    assert {item.unique_name for item in pending_items} == {
        "t_status.PENDING_0",
        "t_status.PENDING_1",
    }

    proven_items = list(
        kb_storage.get_items_by_status(ItemStatus.PROVEN, db_path=test_db)
    )
    assert len(proven_items) == 1, "Expected 1 PROVEN item"
    assert proven_items[0].unique_name == "t_status.PROVEN_1"

    failed_items = list(
        kb_storage.get_items_by_status(
            ItemStatus.LEAN_VALIDATION_FAILED, db_path=test_db
        )
    )
    assert len(failed_items) == 1
    assert failed_items[0].unique_name == "t_status.LEAN_VALIDATION_FAILED_2"

    accepted_items = list(
        kb_storage.get_items_by_status(ItemStatus.LATEX_ACCEPTED, db_path=test_db)
    )
    assert len(accepted_items) == 1
    assert accepted_items[0].unique_name == "t_status.LATEX_ACCEPTED_3"

    # Test retrieving a status with no items
    error_items = list(
        kb_storage.get_items_by_status(ItemStatus.ERROR, db_path=test_db)
    )
    assert len(error_items) == 0, "Expected 0 ERROR items"


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_get_items_by_topic(test_db):
    """Verify retrieving items filtered by topic prefix."""
    # Save items with different topics
    await kb_storage.save_kb_item(
        KBItem(unique_name="t_topic.core.types", topic="Core.Types"),
        client=None,
        db_path=test_db,
    )
    await kb_storage.save_kb_item(
        KBItem(unique_name="t_topic.core.nat", topic="Core.Nat"),
        client=None,
        db_path=test_db,
    )
    await kb_storage.save_kb_item(
        KBItem(unique_name="t_topic.algebra.groups", topic="Algebra.Groups"),
        client=None,
        db_path=test_db,
    )
    await kb_storage.save_kb_item(
        KBItem(unique_name="t_topic.core", topic="Core"), client=None, db_path=test_db
    )
    await kb_storage.save_kb_item(
        KBItem(unique_name="t_topic.analysis.calculus", topic="Analysis.Calculus"),
        client=None,
        db_path=test_db,
    )

    # Test retrieving by prefix "Core" (should match Core, Core.Types, Core.Nat)
    core_items = list(kb_storage.get_items_by_topic("Core", db_path=test_db))
    core_names = {item.unique_name for item in core_items}
    assert len(core_items) == 3
    assert core_names == {"t_topic.core.types", "t_topic.core.nat", "t_topic.core"}

    # Test retrieving by prefix "Core.Nat" (should match only one)
    core_nat_items = list(kb_storage.get_items_by_topic("Core.Nat", db_path=test_db))
    assert len(core_nat_items) == 1
    assert core_nat_items[0].unique_name == "t_topic.core.nat"

    # Test retrieving by prefix "Algebra"
    algebra_items = list(kb_storage.get_items_by_topic("Algebra", db_path=test_db))
    assert len(algebra_items) == 1
    assert algebra_items[0].unique_name == "t_topic.algebra.groups"

    # Test retrieving by prefix not matching any topic
    calculus_items = list(
        kb_storage.get_items_by_topic("Calculus", db_path=test_db)
    )  # Note: Case-sensitive
    assert len(calculus_items) == 0

    # Test retrieving all items (empty prefix)
    all_items = list(kb_storage.get_items_by_topic("", db_path=test_db))
    assert len(all_items) == 5


def test_retrieve_non_existent(test_db):
    """Verify retrieving non-existent items by ID or name returns None."""
    retrieved_id = kb_storage.get_kb_item_by_id(99999, db_path=test_db)
    assert retrieved_id is None, (
        "get_kb_item_by_id should return None for non-existent ID"
    )

    retrieved_name = kb_storage.get_kb_item_by_name(
        "test.non_existent.name", db_path=test_db
    )
    assert retrieved_name is None, (
        "get_kb_item_by_name should return None for non-existent name"
    )


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_save_and_retrieve_embeddings(test_db):
    """Verify saving and retrieving manually set embedding BLOB data."""
    # Create dummy embedding data
    embedding_data_latex = np.array([0.1, -0.2, 0.3], dtype=EMBEDDING_DTYPE).tobytes()
    embedding_data_nl = np.array([0.4, 0.5, -0.6], dtype=EMBEDDING_DTYPE).tobytes()

    item = KBItem(
        unique_name="test.with.embeddings",
        item_type=ItemType.EXAMPLE,
        latex_statement="Example statement",
        latex_proof="Example proof",  # Example can have proof
        embedding_latex=embedding_data_latex,  # Set manually
        embedding_nl=embedding_data_nl,  # Set manually
    )
    # Save without client - this should preserve the manually set embeddings
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)

    # Assertions after save
    assert saved_item.id is not None
    assert saved_item.embedding_latex == embedding_data_latex, (
        "Manually set latex embedding was overwritten/lost"
    )
    assert saved_item.embedding_nl == embedding_data_nl, (
        "Manually set nl embedding was overwritten/lost"
    )

    # Retrieve and verify stored blobs
    retrieved_item = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved_item is not None
    assert retrieved_item.embedding_latex == embedding_data_latex, (
        "Retrieved latex embedding blob mismatch"
    )
    assert retrieved_item.embedding_nl == embedding_data_nl, (
        "Retrieved nl embedding blob mismatch"
    )


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_complex_dependencies(test_db):
    """Verify saving and retrieving item with a list of Lean dependencies."""
    deps = [
        "Core.Init.Default",
        "Algebra.Group.Defs",
        "Topology.Basic",
        "MyProject.Module.Sub",
    ]
    item = KBItem(
        unique_name="test.complex.deps", dependencies=deps
    )  # Use default type/status etc.
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    assert saved_item.id is not None

    retrieved_item = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved_item is not None
    assert retrieved_item.dependencies == deps, (
        "Retrieved Lean dependencies list mismatch"
    )


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_complex_latex_links(test_db):
    """Verify saving and retrieving item with multiple complex LatexLink objects."""
    links = [
        LatexLink(
            citation_text="S1", link_type="definition", source_identifier="DOI:X"
        ),
        LatexLink(
            citation_text="S2",
            link_type="proof",
            source_identifier="URL:Y",
            verified_by_human=True,
        ),
        LatexLink(
            citation_text="S3",
            link_type="statement",
            source_identifier=None,
            latex_snippet="x=y",
            match_confidence=0.9,
        ),
    ]
    item = KBItem(unique_name="test.complex.links", latex_links=links)
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    assert saved_item.id is not None

    retrieved_item = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved_item is not None
    assert len(retrieved_item.latex_links) == len(links), (
        "Incorrect number of LatexLinks retrieved"
    )

    # Compare lists of dataclasses for equality (assumes __eq__ is default or defined correctly)
    assert retrieved_item.latex_links == links, (
        "Retrieved LatexLinks list content mismatch"
    )


# --- New Tests for Plan Dependencies and Failure Count ---


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_add_plan_dependency(test_db):
    """Verify saving, adding, and retrieving plan_dependencies."""
    item = KBItem(unique_name="test.plan.dep.target")
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    await asyncio.sleep(0.01)

    # Add dependencies and check updates
    saved_item.add_plan_dependency("plan.dep.1")
    updated_item1 = await kb_storage.save_kb_item(
        saved_item, client=None, db_path=test_db
    )
    assert updated_item1.plan_dependencies == ["plan.dep.1"]

    # Add same again (should not duplicate)
    updated_item1.add_plan_dependency("plan.dep.1")
    updated_item2 = await kb_storage.save_kb_item(
        updated_item1, client=None, db_path=test_db
    )
    assert updated_item2.plan_dependencies == ["plan.dep.1"]

    # Add second dependency
    updated_item2.add_plan_dependency("plan.dep.2")
    updated_item3 = await kb_storage.save_kb_item(
        updated_item2, client=None, db_path=test_db
    )
    assert updated_item3.plan_dependencies == ["plan.dep.1", "plan.dep.2"]

    # Retrieve final state and verify
    retrieved = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved is not None
    assert retrieved.plan_dependencies == ["plan.dep.1", "plan.dep.2"]


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_increment_failure_count(test_db):
    """Verify saving, incrementing, and retrieving failure_count."""
    item = KBItem(unique_name="test.failure.count")
    saved_item = await kb_storage.save_kb_item(item, client=None, db_path=test_db)
    assert saved_item.failure_count == 0, "Initial failure_count should be 0"
    await asyncio.sleep(0.01)

    # Increment and save multiple times
    saved_item.increment_failure_count()
    updated_item1 = await kb_storage.save_kb_item(
        saved_item, client=None, db_path=test_db
    )
    assert updated_item1.failure_count == 1

    updated_item1.increment_failure_count()
    updated_item2 = await kb_storage.save_kb_item(
        updated_item1, client=None, db_path=test_db
    )
    assert updated_item2.failure_count == 2

    # Retrieve final state and verify
    retrieved = kb_storage.get_kb_item_by_id(saved_item.id, db_path=test_db)
    assert retrieved is not None
    assert retrieved.failure_count == 2


# Mark tests that call async functions
@pytest.mark.asyncio
async def test_item_type_requires_proof_handling(test_db):
    """Verify latex_proof is None when ItemType does not require proof."""
    # Test case 1: Theorem (requires proof)
    item_thm = KBItem(
        unique_name="test.proof.required",
        item_type=ItemType.THEOREM,
        latex_proof="Proof content",
    )
    saved_thm = await kb_storage.save_kb_item(item_thm, client=None, db_path=test_db)
    retrieved_thm = kb_storage.get_kb_item_by_id(saved_thm.id, db_path=test_db)
    assert retrieved_thm.latex_proof == "Proof content", (
        "Proof should be saved for Theorem"
    )

    # Test case 2: Definition (does not require proof)
    item_def = KBItem(
        unique_name="test.proof.not_required",
        item_type=ItemType.DEFINITION,
        latex_proof="Should be ignored",
    )
    # Check __post_init__ or save logic nullifies it
    assert item_def.latex_proof is None, (
        "Proof should be None after __post_init__ for Definition"
    )
    saved_def = await kb_storage.save_kb_item(item_def, client=None, db_path=test_db)
    retrieved_def = kb_storage.get_kb_item_by_id(saved_def.id, db_path=test_db)
    assert retrieved_def.latex_proof is None, (
        "Proof should be None in DB for Definition"
    )

    # Test case 3: Update item type to one not requiring proof
    retrieved_thm.item_type = ItemType.REMARK  # Change to Remark
    retrieved_thm.latex_proof = "This should also be ignored now"
    updated_thm = await kb_storage.save_kb_item(
        retrieved_thm, client=None, db_path=test_db
    )
    retrieved_remark = kb_storage.get_kb_item_by_id(updated_thm.id, db_path=test_db)
    assert retrieved_remark.item_type == ItemType.REMARK
    assert retrieved_remark.latex_proof is None, (
        "Proof should become None after type change and save"
    )
