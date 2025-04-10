# File: tests/unit/kb/test_storage_unit.py

import enum  # Needed for dummy classes if import fails below
import json
import os
import sys
import time  # Import time for sleep
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np  # For embedding tests
import pytest

# --- Add project root to allow importing 'src' ---
# This assumes the tests are run from the project root directory (e.g., using pytest)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

try:
    # Assuming pytest runs from root and pytest.ini sets pythonpath=src
    from dataclasses import dataclass, field  # Added missing imports
    from typing import (
        Any,
        Dict,
        Generator,
        List,
        Optional,
        Tuple,
    )  # Added missing imports

    from lean_automator.kb.storage import (
        EMBEDDING_DTYPE,
        ItemStatus,
        ItemType,
        KBItem,
        LatexLink,
        _sentinel,
    )
except ImportError as e:
    print(f"Import failed: {e}. Ensure tests are run from the project root.")
    # Define dummies if needed for basic loading, but tests will likely fail
    # Define dummy imports used in the dummy KBItem
    from dataclasses import dataclass, field
    from typing import List, Optional

    class ItemType(enum.Enum):
        THEOREM = 1
        LEMMA = 2
        EXAMPLE = 3
        AXIOM = 4
        PROPOSITION = 5  # dummy

    class ItemStatus(enum.Enum):
        PENDING = 1
        PROVEN = 2
        ERROR = 3
        LEAN_VALIDATION_FAILED = 4
        LATEX_ACCEPTED = 5
        LEAN_VALIDATION_IN_PROGRESS = 6  # dummy

    @dataclass
    class LatexLink:
        citation_text: str = ""
        link_type: str = "statement"
        source_identifier: Optional[str] = None
        latex_snippet: Optional[str] = None
        match_confidence: Optional[float] = None
        verified_by_human: bool = False

    @dataclass
    class KBItem:
        id: Optional[int] = None
        unique_name: str = ""
        item_type: ItemType = ItemType.THEOREM
        description_nl: str = ""
        latex_statement: Optional[str] = None
        latex_proof: Optional[str] = None
        lean_code: str = ""
        embedding_nl: Optional[bytes] = None
        embedding_latex: Optional[bytes] = None
        topic: str = "General"
        plan_dependencies: List[str] = field(default_factory=list)
        dependencies: List[str] = field(default_factory=list)
        latex_links: List[LatexLink] = field(default_factory=list)
        status: ItemStatus = ItemStatus.PENDING
        failure_count: int = 0
        latex_review_feedback: Optional[str] = None
        generation_prompt: Optional[str] = None
        raw_ai_response: Optional[str] = None
        lean_error_log: Optional[str] = None
        created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
        last_modified_at: datetime = field(
            default_factory=lambda: datetime.now(timezone.utc)
        )

        def __post_init__(self):
            pass

        def update_status(self, *args, **kwargs):
            self.last_modified_at = datetime.now(timezone.utc)

        def add_plan_dependency(self, *args, **kwargs):
            self.last_modified_at = datetime.now(timezone.utc)

        def add_dependency(self, *args, **kwargs):
            self.last_modified_at = datetime.now(timezone.utc)

        def add_latex_link(self, *args, **kwargs):
            self.last_modified_at = datetime.now(timezone.utc)

        def increment_failure_count(self):
            self.failure_count += 1
            self.last_modified_at = datetime.now(timezone.utc)

        # update_olean method removed
        def to_dict_for_db(self):
            return asdict(self)  # Simplified for dummy

        @classmethod
        def from_db_dict(cls, data):
            return cls(**data)  # Simplified for dummy

        # Add dummy requires_proof method to ItemType if needed by KBItem dummy
        def requires_proof(self) -> bool:
            return self in {
                ItemType.THEOREM,
                ItemType.LEMMA,
                ItemType.PROPOSITION,
                ItemType.EXAMPLE,
            }

        ItemType.requires_proof = requires_proof

    _sentinel = object()
    EMBEDDING_DTYPE = np.float32


# --- Tests for LatexLink ---
# (Assuming these passed before and need no changes)


def test_latex_link_defaults():
    """Test default values when creating a LatexLink."""
    link = LatexLink(citation_text="Test Citation")
    assert link.citation_text == "Test Citation"
    assert link.link_type == "statement"  # Default value
    assert link.source_identifier is None
    assert link.latex_snippet is None
    assert link.match_confidence is None
    assert link.verified_by_human is False  # Default value


def test_latex_link_all_fields():
    """Test creating a LatexLink with all fields specified."""
    link = LatexLink(
        citation_text="Book X, Thm 1",
        link_type="proof",
        source_identifier="ISBN:123",
        latex_snippet="\\begin{proof}...",
        match_confidence=0.95,
        verified_by_human=True,
    )
    assert link.citation_text == "Book X, Thm 1"
    assert link.link_type == "proof"
    assert link.source_identifier == "ISBN:123"
    assert link.latex_snippet == "\\begin{proof}..."
    assert link.match_confidence == 0.95
    assert link.verified_by_human is True


def test_latex_link_from_dict_basic():
    """Test LatexLink.from_dict with minimal data."""
    data = {"citation_text": "Paper Y, Lemma 2"}
    link = LatexLink.from_dict(data)
    assert link.citation_text == "Paper Y, Lemma 2"
    assert link.link_type == "statement"  # Should default
    assert link.source_identifier is None


def test_latex_link_from_dict_full():
    """Test LatexLink.from_dict with all fields."""
    data = {
        "citation_text": "Book X, Thm 1",
        "link_type": "proof",
        "source_identifier": "ISBN:123",
        "latex_snippet": "\\begin{proof}...",
        "match_confidence": 0.95,
        "verified_by_human": True,
    }
    link = LatexLink.from_dict(data)
    assert link.citation_text == data["citation_text"]
    assert link.link_type == data["link_type"]
    assert link.source_identifier == data["source_identifier"]
    assert link.latex_snippet == data["latex_snippet"]
    assert link.match_confidence == data["match_confidence"]
    assert link.verified_by_human == data["verified_by_human"]


def test_latex_link_from_dict_missing_link_type():
    """Test LatexLink.from_dict defaults link_type if missing."""
    data = {"citation_text": "Source Z"}
    link = LatexLink.from_dict(data)
    assert link.link_type == "statement"


def test_latex_link_from_dict_invalid_key():
    """Test LatexLink.from_dict raises TypeError for unexpected keys."""
    data = {"citation_text": "Source Z", "unexpected_field": 123}
    with pytest.raises(TypeError):
        LatexLink.from_dict(data)


# --- Tests for KBItem ---


def test_kbitem_defaults():
    """Test the default values when creating a KBItem."""
    item = KBItem()
    assert item.id is None
    assert len(item.unique_name) > 10
    try:
        UUID(item.unique_name.replace("item_", ""), version=4)
    except ValueError:
        pytest.fail(f"Default unique_name '{item.unique_name}' is not UUID based.")
    assert item.item_type == ItemType.THEOREM
    assert item.latex_statement is None  # Changed from latex_exposition
    assert item.latex_proof is None  # Check default proof is None
    assert item.lean_code == ""
    assert item.description_nl == ""
    # assert item.lean_olean is None # Removed lean_olean check
    assert item.embedding_latex is None
    assert item.embedding_nl is None
    assert item.topic == "General"
    assert item.plan_dependencies == []
    assert item.dependencies == []
    assert item.latex_links == []
    assert item.status == ItemStatus.PENDING
    assert item.failure_count == 0
    assert item.generation_prompt is None
    assert item.raw_ai_response is None
    assert item.lean_error_log is None
    now_utc = datetime.now(timezone.utc)
    assert isinstance(item.created_at, datetime)
    assert item.created_at.tzinfo == timezone.utc
    assert (now_utc - item.created_at) < timedelta(seconds=5)
    assert isinstance(item.last_modified_at, datetime)
    assert item.last_modified_at.tzinfo == timezone.utc
    assert (now_utc - item.last_modified_at) < timedelta(seconds=5)


def test_kbitem_post_init_validations():
    """Test __post_init__ validations."""
    with pytest.raises(ValueError, match="unique_name cannot be empty"):
        KBItem(unique_name="")

    naive_dt = datetime(2025, 1, 1, 12, 0, 0)
    item = KBItem(created_at=naive_dt, last_modified_at=naive_dt)
    assert item.created_at.tzinfo == timezone.utc
    assert item.last_modified_at.tzinfo == timezone.utc

    item = KBItem(lean_code=123)  # Test type conversion
    assert item.lean_code == "123"

    # Test latex_proof is None if not required
    item_def = KBItem(item_type=ItemType.DEFINITION, latex_proof="Should be removed")
    assert item_def.latex_proof is None
    item_thm = KBItem(item_type=ItemType.THEOREM, latex_proof="Should remain")
    assert item_thm.latex_proof == "Should remain"


def test_kbitem_update_status():
    """Test the update_status method."""
    item = KBItem()
    original_mod_time = item.last_modified_at
    time.sleep(0.01)  # Ensure time progresses

    item.update_status(
        ItemStatus.LEAN_VALIDATION_FAILED, error_log="Lean Compile Error"
    )
    assert item.status == ItemStatus.LEAN_VALIDATION_FAILED
    assert item.lean_error_log == "Lean Compile Error"
    assert item.last_modified_at > original_mod_time

    mod_time_after_fail = item.last_modified_at
    time.sleep(0.01)
    # Test updating status without changing error log (using _sentinel implicitly)
    item.update_status(ItemStatus.ERROR)
    assert item.status == ItemStatus.ERROR
    assert item.lean_error_log == "Lean Compile Error"  # Should not change
    assert item.last_modified_at > mod_time_after_fail

    mod_time_after_error = item.last_modified_at
    time.sleep(0.01)
    # Test updating status and explicitly setting error log to None
    # Use LATEX_ACCEPTED as a valid status instead of LATEX_REVIEW_PASSED
    item.update_status(ItemStatus.LATEX_ACCEPTED, error_log=None)
    assert item.status == ItemStatus.LATEX_ACCEPTED
    assert item.lean_error_log is None
    assert item.last_modified_at > mod_time_after_error

    # Test invalid status type raises error
    with pytest.raises(TypeError):
        item.update_status("NOT_A_STATUS")

    # Test clearing logs when moving to PROVEN
    item.lean_error_log = "Some error"
    item.latex_review_feedback = "Some feedback"
    item.update_status(ItemStatus.PROVEN)
    assert item.lean_error_log is None
    assert item.latex_review_feedback is None

    # Test clearing latex review feedback when moving to LATEX_ACCEPTED
    item.latex_review_feedback = "Some feedback"
    item.update_status(ItemStatus.LATEX_ACCEPTED)
    assert item.latex_review_feedback is None


def test_kbitem_add_plan_dependency():
    """Test adding planning dependencies."""
    item = KBItem()
    original_mod_time = item.last_modified_at
    time.sleep(0.01)

    item.add_plan_dependency("plan_dep1")
    assert item.plan_dependencies == ["plan_dep1"]
    assert item.last_modified_at > original_mod_time

    mod_time_after_add1 = item.last_modified_at
    item.add_plan_dependency("plan_dep1")  # Add same again
    assert item.plan_dependencies == ["plan_dep1"]  # Should not duplicate
    assert item.last_modified_at == mod_time_after_add1  # Time shouldn't update

    time.sleep(0.01)
    item.add_plan_dependency("plan_dep2")
    assert item.plan_dependencies == ["plan_dep1", "plan_dep2"]
    assert item.last_modified_at > mod_time_after_add1


def test_kbitem_add_dependency():
    """Test adding Lean dependencies."""
    item = KBItem()
    original_mod_time = item.last_modified_at
    time.sleep(0.01)

    item.add_dependency("lean_dep1")
    assert item.dependencies == ["lean_dep1"]
    assert item.last_modified_at > original_mod_time

    mod_time_after_add1 = item.last_modified_at
    item.add_dependency("lean_dep1")  # Add same again
    assert item.dependencies == ["lean_dep1"]  # Should not duplicate
    assert item.last_modified_at == mod_time_after_add1  # Time shouldn't update

    time.sleep(0.01)
    item.add_dependency("lean_dep2")
    assert item.dependencies == ["lean_dep1", "lean_dep2"]
    assert item.last_modified_at > mod_time_after_add1


def test_kbitem_add_latex_link():
    """Test adding LaTeX links (external references)."""
    item = KBItem()
    link1 = LatexLink(citation_text="L1")
    link2 = LatexLink(citation_text="L2", link_type="proof")
    original_mod_time = item.last_modified_at
    time.sleep(0.01)

    item.add_latex_link(link1)
    assert item.latex_links == [link1]
    mod_time_after_add1 = item.last_modified_at
    assert mod_time_after_add1 > original_mod_time

    time.sleep(0.01)
    item.add_latex_link(link2)
    assert item.latex_links == [link1, link2]
    assert item.last_modified_at > mod_time_after_add1


def test_kbitem_increment_failure_count():
    """Test incrementing the failure count."""
    item = KBItem()
    assert item.failure_count == 0
    original_mod_time = item.last_modified_at
    time.sleep(0.01)

    item.increment_failure_count()
    assert item.failure_count == 1
    assert item.last_modified_at > original_mod_time

    mod_time_after_inc1 = item.last_modified_at
    time.sleep(0.01)
    item.increment_failure_count()
    assert item.failure_count == 2
    assert item.last_modified_at > mod_time_after_inc1


# def test_kbitem_update_olean(): <-- This entire test function is removed


def test_kbitem_to_dict_for_db():
    """Test serialization to dictionary for database storage."""
    now = datetime.now(timezone.utc)
    plan_deps = ["plan.dep.A", "plan.dep.B"]
    # olean_data = b'olean_content' # Removed - no longer part of KBItem
    # Create dummy embedding data - should NOT be included in the dict FOR DB
    embedding_data_latex = np.array([0.1, 0.2], dtype=EMBEDDING_DTYPE).tobytes()
    embedding_data_nl = np.array([0.3, 0.4], dtype=EMBEDDING_DTYPE).tobytes()

    item = KBItem(
        id=123,
        unique_name="test.serial.item",
        item_type=ItemType.LEMMA,
        latex_statement="Latex for lemma l.",  # Changed from latex_exposition
        latex_proof="Proof for lemma l.",  # Added proof
        lean_code="lemma l : X := Y",
        description_nl="A test lemma.",
        # lean_olean=olean_data, # Removed
        embedding_latex=embedding_data_latex,  # Provided but excluded by to_dict_for_db
        embedding_nl=embedding_data_nl,  # Provided but excluded by to_dict_for_db
        topic="Test.Serialization",
        plan_dependencies=plan_deps,
        dependencies=["depA", "depB"],
        latex_links=[
            LatexLink(citation_text="L1", link_type="statement"),
            LatexLink(citation_text="L2", link_type="proof", source_identifier="ID002"),
        ],
        status=ItemStatus.PROVEN,
        failure_count=2,
        generation_prompt="Generate this",
        raw_ai_response="AI did",
        lean_error_log=None,
        created_at=now - timedelta(hours=1),
        last_modified_at=now,
    )

    db_dict = item.to_dict_for_db()

    assert isinstance(db_dict, dict)
    assert db_dict["id"] == 123
    assert db_dict["unique_name"] == "test.serial.item"
    assert db_dict["item_type"] == "LEMMA"
    assert db_dict["latex_statement"] == "Latex for lemma l."  # Changed key
    assert db_dict["latex_proof"] == "Proof for lemma l."  # Check proof included
    assert db_dict["lean_code"] == "lemma l : X := Y"
    assert db_dict["description_nl"] == "A test lemma."
    # assert db_dict["lean_olean"] == olean_data # Removed lean_olean check
    assert "embedding_latex" not in db_dict  # Check embedding excluded
    assert "embedding_nl" not in db_dict  # Check embedding excluded
    assert db_dict["topic"] == "Test.Serialization"
    assert isinstance(db_dict["plan_dependencies"], str)
    assert json.loads(db_dict["plan_dependencies"]) == plan_deps
    assert isinstance(db_dict["dependencies"], str)
    assert json.loads(db_dict["dependencies"]) == ["depA", "depB"]
    assert isinstance(db_dict["latex_links"], str)
    parsed_links = json.loads(db_dict["latex_links"])
    assert len(parsed_links) == 2
    assert parsed_links[0]["citation_text"] == "L1"
    assert parsed_links[1]["citation_text"] == "L2"
    assert db_dict["status"] == "PROVEN"
    assert db_dict["failure_count"] == 2
    assert db_dict["generation_prompt"] == "Generate this"
    assert db_dict["raw_ai_response"] == "AI did"
    assert db_dict["lean_error_log"] is None
    assert isinstance(db_dict["created_at"], str)
    assert datetime.fromisoformat(db_dict["created_at"]) == item.created_at
    assert isinstance(db_dict["last_modified_at"], str)
    assert datetime.fromisoformat(db_dict["last_modified_at"]) == item.last_modified_at

    # Test case where proof is not required
    item_axiom = KBItem(item_type=ItemType.AXIOM, latex_proof="Should be ignored")
    db_dict_axiom = item_axiom.to_dict_for_db()
    assert db_dict_axiom["latex_proof"] is None


def test_kbitem_from_db_dict():
    """Test deserialization from a database-like dictionary."""
    now_iso = datetime.now(timezone.utc).isoformat()
    yesterday_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    plan_deps = ["plan.dep.C"]
    # olean_data = b'some_olean' # Removed - DB dict won't have this
    embedding_data_latex = np.array([1.1, 1.2], dtype=EMBEDDING_DTYPE).tobytes()
    embedding_data_nl = np.array([1.3, 1.4], dtype=EMBEDDING_DTYPE).tobytes()

    db_dict = {
        "id": 456,
        "unique_name": "test.deserial.item",
        "item_type": "PROPOSITION",  # Requires proof
        "latex_statement": "Prop P requires Q.",  # Changed key
        "latex_proof": "Proof of P.",  # Added proof
        "lean_code": "prop P : Q",
        "description_nl": "Deserialize me",
        # "lean_olean": olean_data, # Removed
        "embedding_latex": embedding_data_latex,  # Embeddings still loaded if present in DB
        "embedding_nl": embedding_data_nl,
        "topic": "Test.Deserialization",
        "plan_dependencies": json.dumps(plan_deps),
        "dependencies": json.dumps(["depC"]),
        "latex_links": json.dumps(
            [
                {
                    "citation_text": "L3",
                    "link_type": "statement",
                    "verified_by_human": True,
                }
            ]
        ),
        "status": "LATEX_ACCEPTED",  # Changed from LATEX_REVIEW_PASSED
        "failure_count": 1,
        "generation_prompt": None,
        "raw_ai_response": "Response",
        "lean_error_log": "Error msg",
        "created_at": yesterday_iso,
        "last_modified_at": now_iso,
    }

    item = KBItem.from_db_dict(db_dict)

    assert isinstance(item, KBItem)
    assert item.id == 456
    assert item.unique_name == "test.deserial.item"
    assert item.item_type == ItemType.PROPOSITION
    assert item.latex_statement == "Prop P requires Q."  # Changed assertion key
    assert item.latex_proof == "Proof of P."  # Check proof loaded
    assert item.lean_code == "prop P : Q"
    assert item.description_nl == "Deserialize me"
    # assert item.lean_olean == olean_data # Removed lean_olean check
    assert item.embedding_latex == embedding_data_latex
    assert item.embedding_nl == embedding_data_nl
    assert item.topic == "Test.Deserialization"
    assert item.plan_dependencies == plan_deps
    assert item.dependencies == ["depC"]
    assert isinstance(item.latex_links, list) and len(item.latex_links) == 1
    assert isinstance(item.latex_links[0], LatexLink)
    assert item.latex_links[0].citation_text == "L3"
    assert item.status == ItemStatus.LATEX_ACCEPTED  # Changed assertion value
    assert item.failure_count == 1
    assert item.generation_prompt is None
    assert item.raw_ai_response == "Response"
    assert item.lean_error_log == "Error msg"
    assert item.created_at == datetime.fromisoformat(yesterday_iso)
    assert item.last_modified_at == datetime.fromisoformat(now_iso)

    # Test deserialization where proof is None because type doesn't require it
    db_dict_axiom = db_dict.copy()
    db_dict_axiom["item_type"] = "AXIOM"
    db_dict_axiom["latex_proof"] = "Some proof that should be ignored"
    item_axiom = KBItem.from_db_dict(db_dict_axiom)
    assert item_axiom.item_type == ItemType.AXIOM
    assert item_axiom.latex_proof is None  # Verify it was set to None


def test_kbitem_from_db_dict_minimal():
    """Test deserialization with only mandatory fields and missing optionals/empty lists."""
    now_iso = datetime.now(timezone.utc).isoformat()
    db_dict = {
        "unique_name": "test.deserial.minimal",
        "item_type": "AXIOM",
        "lean_code": "",  # Minimal needs empty string based on class def.
        "status": ItemStatus.PROVEN.name,  # Axioms are PROVEN
        "created_at": now_iso,
        "last_modified_at": now_iso,
        # All other Optional fields are missing
    }
    item = KBItem.from_db_dict(db_dict)

    assert item.id is None
    assert item.unique_name == "test.deserial.minimal"
    assert item.item_type == ItemType.AXIOM
    assert item.latex_statement is None  # Changed from latex_exposition
    assert item.latex_proof is None
    assert item.lean_code == ""
    assert item.description_nl == ""
    # assert item.lean_olean is None # Removed lean_olean check
    assert item.embedding_latex is None
    assert item.embedding_nl is None
    assert item.topic == "General"
    assert item.plan_dependencies == []
    assert item.dependencies == []
    assert item.latex_links == []
    assert item.status == ItemStatus.PROVEN
    assert item.failure_count == 0
    assert item.generation_prompt is None
    assert item.raw_ai_response is None
    assert item.lean_error_log is None
    assert item.created_at == datetime.fromisoformat(now_iso)
    assert item.last_modified_at == datetime.fromisoformat(now_iso)


def test_kbitem_from_db_dict_error_handling():
    """Test error handling during deserialization."""
    now_iso = datetime.now(timezone.utc).isoformat()
    # Changed status to a valid one based on previous error message hint
    base_dict = {
        "unique_name": "test.error",
        "item_type": "THEOREM",
        "lean_code": "",
        "status": "LEAN_VALIDATION_IN_PROGRESS",
        "created_at": now_iso,
        "last_modified_at": now_iso,
    }

    # Test valid status
    item = KBItem.from_db_dict(base_dict)
    assert item.status == ItemStatus.LEAN_VALIDATION_IN_PROGRESS  # Changed assertion

    # Missing mandatory field (unique_name)
    with pytest.raises(
        ValueError, match="Error deserializing KBItem .* from DB dict: 'unique_name'"
    ):
        bad_dict_missing_field = base_dict.copy()
        del bad_dict_missing_field["unique_name"]
        KBItem.from_db_dict(bad_dict_missing_field)

    # Invalid JSON for plan_dependencies
    with pytest.raises(
        ValueError, match="Error deserializing KBItem .* from DB dict: Expecting value"
    ):
        bad_dict_invalid_json = base_dict.copy()
        bad_dict_invalid_json["plan_dependencies"] = "[Invalid JSON"
        KBItem.from_db_dict(bad_dict_invalid_json)

    # Invalid Enum value for Status - should default to PENDING with a warning
    bad_dict_invalid_status = base_dict.copy()
    bad_dict_invalid_status["status"] = "INVALID_STATUS_XYZ"
    # Use context manager style for warns
    with pytest.warns(UserWarning, match="Invalid status 'INVALID_STATUS_XYZ'"):
        item_invalid_status = KBItem.from_db_dict(bad_dict_invalid_status)
    assert item_invalid_status.status == ItemStatus.PENDING  # Assert the default status

    # Test the default fallback for status if key is missing
    bad_dict_missing_status = base_dict.copy()
    del bad_dict_missing_status["status"]
    item_missing_status = KBItem.from_db_dict(bad_dict_missing_status)
    assert item_missing_status.status == ItemStatus.PENDING  # Check default status used

    # Invalid Enum value for Type (This should still raise ValueError wrapping KeyError)
    with pytest.raises(
        ValueError, match="Error deserializing KBItem .* from DB dict: 'INVALID_TYPE'"
    ):
        bad_dict_invalid_type = base_dict.copy()
        bad_dict_invalid_type["item_type"] = "INVALID_TYPE"
        KBItem.from_db_dict(bad_dict_invalid_type)

    # Invalid Date format
    with pytest.raises(
        ValueError,
        match="Error deserializing KBItem .* from DB dict: Invalid isoformat string",
    ):
        bad_dict_invalid_date = base_dict.copy()
        bad_dict_invalid_date["created_at"] = "Not a date"
        KBItem.from_db_dict(bad_dict_invalid_date)

    # Invalid type for failure_count (Should now warn and default to 0)
    bad_dict_bad_count = base_dict.copy()
    bad_dict_bad_count["failure_count"] = "not a number"
    with pytest.warns(UserWarning, match="Invalid type for failure_count"):
        item_bad_count = KBItem.from_db_dict(bad_dict_bad_count)
    assert item_bad_count.failure_count == 0


def test_serialization_round_trip():
    """Test that an item can be serialized and deserialized back to an equivalent object."""
    now = datetime.now(timezone.utc)
    plan_deps = ["plan.dep.X"]
    # olean_data = b'olean_for_round_trip' # Removed
    embedding_data_latex = np.array([2.1, 2.2], dtype=EMBEDDING_DTYPE).tobytes()
    embedding_data_nl = np.array([2.3, 2.4], dtype=EMBEDDING_DTYPE).tobytes()

    item_original = KBItem(
        unique_name="test.roundtrip.item",
        item_type=ItemType.EXAMPLE,
        latex_statement="Example LaTeX statement.",  # Changed from latex_exposition
        latex_proof="Example LaTeX proof.",  # Added proof for EXAMPLE type
        lean_code="example : 1 + 1 = 2 := rfl",
        description_nl="Round trip test",
        # lean_olean=olean_data, # Removed
        embedding_latex=embedding_data_latex,  # Provide initial embeddings
        embedding_nl=embedding_data_nl,
        topic="Test.RoundTrip",
        plan_dependencies=plan_deps,
        dependencies=["depX"],
        latex_links=[LatexLink(citation_text="L4")],
        status=ItemStatus.PROVEN,  # EXAMPLE type requires proof -> PROVEN status
        failure_count=3,
        generation_prompt="Prompt",
        raw_ai_response="Response",
        lean_error_log=None,  # Explicitly None
        latex_review_feedback=None,  # Explicitly None
        created_at=now - timedelta(minutes=5),
        last_modified_at=now,
    )

    # Simulate DB save/load cycle
    # 1. Convert to dict for DB (excludes embeddings)
    db_dict_for_save = item_original.to_dict_for_db()
    # ID is handled by DB, remove if present (shouldn't be for new item)
    db_dict_for_save.pop("id", None)

    # 2. Simulate DB storing the data, including fields NOT in db_dict_for_save
    # Create the dictionary as it would look when read *from* the DB
    # Assume DB assigned ID 999
    db_dict_from_load = db_dict_for_save.copy()
    db_dict_from_load["id"] = 999
    # Add back the embeddings as they would be stored separately and loaded
    db_dict_from_load["embedding_latex"] = item_original.embedding_latex
    db_dict_from_load["embedding_nl"] = item_original.embedding_nl

    # 3. Deserialize from the DB-like dictionary
    item_reloaded = KBItem.from_db_dict(db_dict_from_load)

    # Assert equivalence field by field
    assert item_reloaded.id == 999  # Check assigned ID
    assert item_reloaded.unique_name == item_original.unique_name
    assert item_reloaded.item_type == item_original.item_type
    assert (
        item_reloaded.latex_statement == item_original.latex_statement
    )  # Changed assertion key
    assert item_reloaded.latex_proof == item_original.latex_proof  # Check proof field
    assert item_reloaded.lean_code == item_original.lean_code
    assert item_reloaded.description_nl == item_original.description_nl
    # assert item_reloaded.lean_olean == item_original.lean_olean # Removed lean_olean check
    assert item_reloaded.embedding_latex == item_original.embedding_latex
    assert item_reloaded.embedding_nl == item_original.embedding_nl
    assert item_reloaded.topic == item_original.topic
    assert item_reloaded.plan_dependencies == item_original.plan_dependencies
    assert item_reloaded.dependencies == item_original.dependencies
    assert len(item_reloaded.latex_links) == len(item_original.latex_links)
    if item_reloaded.latex_links:
        # Compare as dicts for simplicity if order doesn't matter or only one element
        assert asdict(item_reloaded.latex_links[0]) == asdict(
            item_original.latex_links[0]
        )
    assert item_reloaded.status == item_original.status
    assert item_reloaded.failure_count == item_original.failure_count
    assert item_reloaded.generation_prompt == item_original.generation_prompt
    assert item_reloaded.raw_ai_response == item_original.raw_ai_response
    assert item_reloaded.lean_error_log == item_original.lean_error_log
    assert item_reloaded.latex_review_feedback == item_original.latex_review_feedback
    # Compare timestamps with tolerance for float precision
    assert abs(item_reloaded.created_at - item_original.created_at) < timedelta(
        milliseconds=1
    )
    assert abs(
        item_reloaded.last_modified_at - item_original.last_modified_at
    ) < timedelta(milliseconds=1)
