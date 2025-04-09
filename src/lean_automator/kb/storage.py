# File: lean_automator/kb/storage.py

"""Defines data structures and SQLite storage for a mathematical knowledge base.

This module provides the core data structures (`KBItem`, `LatexLink`, etc.)
representing mathematical concepts and their metadata within a knowledge base.
It also includes functions for creating, initializing, saving, and retrieving
these items from an SQLite database. The default database path is determined
by the `KB_DB_PATH` environment variable or defaults to 'knowledge_base.sqlite'.
Functionality includes handling text embeddings (stored as BLOBs) associated
with natural language descriptions and LaTeX statements.
"""

import enum
import uuid
import sqlite3
import json
import warnings
import os
import asyncio
import logging
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Generator, Tuple

try:
    from lean_automator.config.loader import APP_CONFIG
except ImportError:
    warnings.warn("config_loader.APP_CONFIG not found. Default settings may be used.", ImportWarning)
    APP_CONFIG = {} # Provide an empty dict as a fallback

# Use absolute imports assuming 'src' is in the path or project is installed
try:
    from lean_automator.llm.caller import GeminiClient
except ImportError:
    warnings.warn("llm_call.GeminiClient not found. Embedding generation in save_kb_item will fail.", ImportWarning)
    GeminiClient = None # type: ignore


# --- Logging ---
logger = logging.getLogger(__name__)

# --- Sentinel object ---
_sentinel = object()

# --- Database Configuration ---
DEFAULT_DB_PATH = APP_CONFIG.get('database', {}).get('kb_db_path', 'knowledge_base.sqlite')
EMBEDDING_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBEDDING_DTYPE = np.float32

# --- Controlled Vocabularies ---

class ItemType(enum.Enum):
    """Enumerates the different types of items stored in the Knowledge Base."""
    DEFINITION = "Definition"
    AXIOM = "Axiom"
    THEOREM = "Theorem"
    LEMMA = "Lemma"
    PROPOSITION = "Proposition"
    COROLLARY = "Corollary"
    EXAMPLE = "Example"
    REMARK = "Remark"
    CONJECTURE = "Conjecture"
    NOTATION = "Notation"
    STRUCTURE = "Structure"

    def requires_proof(self) -> bool:
        """Checks if this item type typically requires a proof.

        Returns:
            bool: True if the item type is one that generally needs a proof
            (Theorem, Lemma, Proposition, Corollary, Example), False otherwise.
        """
        return self in {
            ItemType.THEOREM, ItemType.LEMMA, ItemType.PROPOSITION,
            ItemType.COROLLARY, ItemType.EXAMPLE # Examples might need verification/proof
        }

class ItemStatus(enum.Enum):
    """Enumerates the possible states of an item in the Knowledge Base lifecycle."""
    # Initial/General States
    PENDING = "Pending" # Default initial state, ready for any processing
    ERROR = "Error"     # General error state

    # LaTeX Processing States
    PENDING_LATEX = "PendingLatex"                      # Ready for LaTeX generation
    LATEX_GENERATION_IN_PROGRESS = "LatexGenInProgress" # LLM is generating LaTeX
    PENDING_LATEX_REVIEW = "PendingLatexReview"         # LaTeX generated, awaiting review
    LATEX_REVIEW_IN_PROGRESS = "LatexReviewInProgress"  # LLM is reviewing LaTeX
    LATEX_ACCEPTED = "LatexAccepted"                    # LaTeX generated and reviewer accepted
    LATEX_REJECTED_FINAL = "LatexRejectedFinal"         # LaTeX rejected after max review cycles

    # Lean Processing States
    PENDING_LEAN = "PendingLean"                        # Ready for Lean code generation (assumes LATEX_ACCEPTED?)
    LEAN_GENERATION_IN_PROGRESS = "LeanGenInProgress"   # LLM is generating Lean code
    LEAN_VALIDATION_PENDING = "LeanValidationPending"   # Lean generated, awaiting check_and_compile
    LEAN_VALIDATION_IN_PROGRESS = "LeanValidationInProgress" # check_and_compile is running
    LEAN_VALIDATION_FAILED = "LeanValidationFailed"     # check_and_compile failed
    PROVEN = "Proven"                                   # Lean code successfully validated (Implies LATEX_ACCEPTED)

    # Specific Proven States (Map to PROVEN for simplicity in logic, but retain for potential filtering)
    AXIOM_ACCEPTED = "Proven" # Axioms are accepted, not strictly proven
    DEFINITION_ADDED = "Proven" # Definitions are added
    REMARK_ADDED = "Proven"
    NOTATION_ADDED = "Proven"
    STRUCTURE_ADDED = "Proven"
    # EXAMPLE_VERIFIED = "Proven" # Maybe example needs proof, keep PROVEN as main state

    # Other States
    CONJECTURE_STATED = "Pending" # Conjectures start as Pending, awaiting proof attempt


# --- Data Structures ---

@dataclass
class LatexLink:
    """Represents a link to a specific component in an external LaTeX source.

    Attributes:
        citation_text (str): The text used for citing the external source (e.g., "[Knuth73]").
        link_type (str): The type of link, defaulting to 'statement'. Other potential
            values could be 'proof', 'definition', etc.
        source_identifier (Optional[str]): A unique identifier for the external
            source document (e.g., DOI, arXiv ID, internal book code).
        latex_snippet (Optional[str]): A short snippet of the LaTeX code from the
            external source related to this link.
        match_confidence (Optional[float]): A confidence score (e.g., 0.0 to 1.0)
            indicating the likelihood that this link correctly points to the intended component.
        verified_by_human (bool): Flag indicating if a human has confirmed the
            correctness of this link. Defaults to False.
    """
    citation_text: str
    link_type: str = 'statement'
    source_identifier: Optional[str] = None
    latex_snippet: Optional[str] = None
    match_confidence: Optional[float] = None
    verified_by_human: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatexLink':
        """Creates a LatexLink instance from a dictionary.

        Sets a default 'link_type' of 'statement' if not present in the input dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing keys corresponding to
                the attributes of LatexLink.

        Returns:
            LatexLink: An instance of the LatexLink class.

        Raises:
            TypeError: If the dictionary keys do not match the LatexLink attributes
                or if the data types are incorrect.
        """
        if 'link_type' not in data: data['link_type'] = 'statement'
        try: return cls(**data)
        except TypeError as e: raise TypeError(f"Error creating LatexLink from dict: {e}") from e

@dataclass
class KBItem:
    """Represents a single node (mathematical statement, definition, etc.) in the Knowledge Base.

    This dataclass holds all information related to a mathematical item, including its
    textual descriptions (NL and LaTeX), formal Lean code, embeddings for semantic search,
    relationships to other items (dependencies), links to external sources, and metadata
    tracking its processing status and history.

    Attributes:
        id (Optional[int]): The primary key identifier from the database. None if not saved yet.
        unique_name (str): A unique human-readable or generated identifier (e.g., "lemma_xyz").
            Defaults to a UUID-based name if not provided.
        item_type (ItemType): The category of the mathematical item (e.g., Theorem, Definition).
        description_nl (str): A natural language description of the item.
        latex_statement (Optional[str]): The LaTeX representation of the item's statement.
        latex_proof (Optional[str]): An informal proof written in LaTeX (only relevant for
            item types that require proof). Set to None otherwise.
        lean_code (str): The formal Lean code representing the item, potentially incomplete (e.g., containing 'sorry').
        embedding_nl (Optional[bytes]): The embedding vector generated from `description_nl`,
            stored as raw bytes (numpy array serialized). None if not generated.
        embedding_latex (Optional[bytes]): The embedding vector generated *only* from
            `latex_statement`, stored as raw bytes. None if not generated.
        topic (str): A general topic or subject area (e.g., "Group Theory", "Measure Theory").
        plan_dependencies (List[str]): A list of unique names of other KBItems identified
            as necessary prerequisites during proof planning or generation.
        dependencies (List[str]): A list of unique names of other KBItems discovered as
            dependencies during Lean code compilation or validation.
        latex_links (List[LatexLink]): A list of links to external LaTeX sources related
            to this item.
        status (ItemStatus): The current processing status of the item in its lifecycle.
        failure_count (int): A counter for the number of times a processing step (like
            validation or generation) has failed for this item. Defaults to 0.
        latex_review_feedback (Optional[str]): Feedback provided by a reviewer if the
            generated LaTeX (statement or proof) was rejected.
        generation_prompt (Optional[str]): The last prompt text sent to an LLM for
            generating content (LaTeX, Lean code, etc.) for this item.
        raw_ai_response (Optional[str]): The last raw text response received from an LLM
            for this item.
        lean_error_log (Optional[str]): Error messages captured from the Lean compiler
            (`check_and_compile`) if validation failed.
        created_at (datetime): Timestamp when the item was first created (UTC).
        last_modified_at (datetime): Timestamp when the item was last modified (UTC).
    """
    # Core Identification
    id: Optional[int] = None
    unique_name: str = field(default_factory=lambda: f"item_{uuid.uuid4().hex}")
    item_type: ItemType = ItemType.THEOREM

    # Content
    description_nl: str = ""
    latex_statement: Optional[str] = None # Contains the corresponding latex statement
    latex_proof: Optional[str] = None     # Added field for informal proof
    lean_code: str = ""                   # Formal Lean code (often with 'sorry')

    # Embeddings (as raw bytes)
    embedding_nl: Optional[bytes] = None      # Generated from description_nl
    embedding_latex: Optional[bytes] = None # Generated ONLY from latex_statement

    # Context & Relationships
    topic: str = "General"
    plan_dependencies: List[str] = field(default_factory=list) # Names of items needed for the proof plan
    dependencies: List[str] = field(default_factory=list) # Names of items discovered during Lean compilation
    latex_links: List[LatexLink] = field(default_factory=list)

    # State & Metadata
    status: ItemStatus = ItemStatus.PENDING
    failure_count: int = 0 # General failure counter
    latex_review_feedback: Optional[str] = None # Store reviewer feedback if LaTeX (statement or proof) rejected
    generation_prompt: Optional[str] = None # Store last prompt sent to LLM (for any generation task)
    raw_ai_response: Optional[str] = None # Store last raw response from LLM
    lean_error_log: Optional[str] = None # Store output from check_and_compile_item on failure
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Performs basic validation and normalization after dataclass initialization.

        Ensures `unique_name` is not empty, converts `lean_code` to string if necessary,
        ensures timestamps are timezone-aware (UTC), and sets `latex_proof` to None
        if the `item_type` does not require a proof.

        Raises:
            ValueError: If `unique_name` is empty.
        """
        if not self.unique_name: raise ValueError("unique_name cannot be empty.")
        if not isinstance(self.lean_code, str): self.lean_code = str(self.lean_code)
        if self.created_at.tzinfo is None: self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.last_modified_at.tzinfo is None: self.last_modified_at = self.last_modified_at.replace(tzinfo=timezone.utc)
        # Ensure latex_proof is None if item type doesn't require proof
        if not self.item_type.requires_proof():
            self.latex_proof = None

    def update_status(self, new_status: ItemStatus, error_log: Optional[str] = _sentinel, review_feedback: Optional[str] = _sentinel):
        """Updates the item's status and optionally clears/sets related logs.

        Also updates the `last_modified_at` timestamp. Error logs and review
        feedback are conditionally cleared when moving to certain "successful"
        states (PROVEN, LATEX_ACCEPTED).

        Args:
            new_status (ItemStatus): The new status to set for the item.
            error_log (Optional[str]): If provided (and not the sentinel), sets the
                `lean_error_log`. Use None to clear it explicitly.
            review_feedback (Optional[str]): If provided (and not the sentinel), sets the
                `latex_review_feedback`. Use None to clear it explicitly.

        Raises:
            TypeError: If `new_status` is not a valid `ItemStatus` enum member.
        """
        if not isinstance(new_status, ItemStatus):
             raise TypeError(f"Invalid status type: {type(new_status)}. Must be ItemStatus enum.")
        self.status = new_status
        if error_log is not _sentinel:
             self.lean_error_log = error_log
        if review_feedback is not _sentinel:
             self.latex_review_feedback = review_feedback

        # Clear logs if moving to a relevant "good" state
        if new_status == ItemStatus.PROVEN:
            self.lean_error_log = None
            self.latex_review_feedback = None # Assume LaTeX was accepted earlier
        if new_status == ItemStatus.LATEX_ACCEPTED:
            self.latex_review_feedback = None

        self.last_modified_at = datetime.now(timezone.utc)

    def add_plan_dependency(self, dep_unique_name: str):
        """Adds a unique name to the list of planning dependencies.

        Updates `last_modified_at` if the dependency was added.

        Args:
            dep_unique_name (str): The unique name of the KBItem dependency.
        """
        if dep_unique_name not in self.plan_dependencies:
            self.plan_dependencies.append(dep_unique_name)
            self.last_modified_at = datetime.now(timezone.utc)

    def add_dependency(self, dep_unique_name: str):
        """Adds a unique name to the list of discovered Lean dependencies.

        Updates `last_modified_at` if the dependency was added.

        Args:
            dep_unique_name (str): The unique name of the KBItem dependency.
        """
        if dep_unique_name not in self.dependencies:
            self.dependencies.append(dep_unique_name)
            self.last_modified_at = datetime.now(timezone.utc)

    def add_latex_link(self, link: LatexLink):
        """Adds a LatexLink object to the item's list of links.

        Updates `last_modified_at`.

        Args:
            link (LatexLink): The LatexLink object to add.
        """
        self.latex_links.append(link)
        self.last_modified_at = datetime.now(timezone.utc)

    def increment_failure_count(self):
        """Increments the general failure counter and updates modification time."""
        self.failure_count += 1
        self.last_modified_at = datetime.now(timezone.utc)

    # --- Serialization ---
    def to_dict_for_db(self) -> Dict[str, Any]:
        """Serializes the KBItem instance into a dictionary suitable for database storage.

        Converts complex types (enums, lists, datetimes, LatexLink list) into
        database-compatible formats (strings, JSON strings). Excludes embedding
        fields, as they are typically handled separately (e.g., direct BLOB updates).
        Sets `latex_proof` to None if the item type does not require one.

        Returns:
            Dict[str, Any]: A dictionary representation of the item ready for DB insertion/update.
        """
        # Ensure proof is None if not required
        proof = self.latex_proof if self.item_type.requires_proof() else None
        # Exclude embeddings here, they are handled separately in save_kb_item
        return {
            "id": self.id,
            "unique_name": self.unique_name,
            "item_type": self.item_type.name, # Store enum name as string
            "description_nl": self.description_nl,
            "latex_statement": self.latex_statement,
            "latex_proof": proof,
            "lean_code": self.lean_code,
            "topic": self.topic,
            "plan_dependencies": json.dumps(self.plan_dependencies), # Serialize list to JSON string
            "dependencies": json.dumps(self.dependencies), # Serialize list to JSON string
            "latex_links": json.dumps([asdict(link) for link in self.latex_links]), # Serialize list of dicts
            "status": self.status.name, # Store enum name as string
            "failure_count": self.failure_count,
            "latex_review_feedback": self.latex_review_feedback,
            "generation_prompt": self.generation_prompt,
            "raw_ai_response": self.raw_ai_response,
            "lean_error_log": self.lean_error_log,
            "created_at": self.created_at.isoformat(), # Store datetime as ISO string
            "last_modified_at": self.last_modified_at.isoformat() # Store datetime as ISO string
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'KBItem':
        """Creates a KBItem instance from a dictionary retrieved from the database.

        Handles deserialization of complex types (enums from names, lists from JSON strings,
        datetimes from ISO strings, LatexLink list from JSON). Includes error handling
        for invalid enum values or JSON decoding errors. Retrieves embedding fields
        directly if present in the input dictionary. Loads `latex_proof` only if the
        determined `item_type` requires it.

        Args:
            data (Dict[str, Any]): A dictionary representing a row fetched from the
                `kb_items` database table, typically obtained via `sqlite3.Row`.
                Must include keys for all necessary KBItem attributes stored in the DB.

        Returns:
            KBItem: An instance of the KBItem class populated with data from the dictionary.

        Raises:
            ValueError: If deserialization fails due to missing keys, invalid enum names,
                JSON decoding errors, or other type mismatches.
        """
        try:
            # Deserialize Enums
            item_type_str = data["item_type"]
            item_type = ItemType[item_type_str] # Raises KeyError if invalid

            status_str = data.get("status", ItemStatus.PENDING.name) # Default if missing
            try:
                status = ItemStatus[status_str]
            except KeyError:
                warnings.warn(f"Invalid status '{status_str}' for item {data.get('unique_name')}. Defaulting to PENDING.")
                status = ItemStatus.PENDING

            # Handle failure_count safely
            failure_count = data.get("failure_count", 0)
            if not isinstance(failure_count, int):
                warnings.warn(f"Invalid type for failure_count: got {type(failure_count)}. Using 0.")
                failure_count = 0

            # Only load latex_proof if the item type requires it
            latex_proof = data.get("latex_proof") if item_type.requires_proof() else None

            return cls(
                id=data.get("id"),
                unique_name=data["unique_name"], # Required key
                item_type=item_type,
                description_nl=data.get("description_nl", ""),
                latex_statement=data.get("latex_statement"),
                latex_proof=latex_proof,
                lean_code=data.get("lean_code", ""),
                # Embeddings are loaded directly as bytes/None if present
                embedding_nl=data.get("embedding_nl"),
                embedding_latex=data.get("embedding_latex"),
                topic=data.get("topic", "General"),
                # Deserialize JSON strings back to lists
                plan_dependencies=json.loads(data.get("plan_dependencies") or '[]'),
                dependencies=json.loads(data.get("dependencies") or '[]'),
                latex_links=[LatexLink.from_dict(link_data) for link_data in json.loads(data.get("latex_links") or '[]')],
                status=status,
                failure_count=failure_count,
                latex_review_feedback=data.get("latex_review_feedback"),
                generation_prompt=data.get("generation_prompt"),
                raw_ai_response=data.get("raw_ai_response"),
                lean_error_log=data.get("lean_error_log"),
                # Deserialize ISO strings back to datetime objects
                created_at=datetime.fromisoformat(data["created_at"]), # Required key
                last_modified_at=datetime.fromisoformat(data["last_modified_at"]) # Required key
             )
        except (KeyError, json.JSONDecodeError, ValueError, TypeError) as e:
             # Catch specific expected errors during deserialization
             raise ValueError(f"Error deserializing KBItem '{data.get('unique_name', 'UNKNOWN_ITEM')}' from DB dict: {e}") from e


# --- Database Interaction Functions ---

def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Establishes and configures an SQLite database connection.

    Uses the provided `db_path` or falls back to the `DEFAULT_DB_PATH`.
    Configures the connection to use `sqlite3.Row` factory for dictionary-like
    row access and enables foreign key constraint enforcement.

    Args:
        db_path (Optional[str]): The path to the SQLite database file. If None,
            uses the path defined by `DEFAULT_DB_PATH`.

    Returns:
        sqlite3.Connection: An configured SQLite database connection object.
    """
    effective_path = db_path if db_path is not None else DEFAULT_DB_PATH
    conn = sqlite3.connect(effective_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    conn.execute("PRAGMA foreign_keys = ON;") # Enforce foreign keys
    return conn

def _add_column_if_not_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str, column_type: str, default_value: Any = None):
    """Adds a column to a table only if it does not already exist. (Internal Helper)

    Checks the table's schema using `PRAGMA table_info` before attempting
    to execute an `ALTER TABLE ADD COLUMN` statement. Handles default values
    for basic types (string, int, float). Logs info on success or warning on failure.

    Args:
        cursor (sqlite3.Cursor): An active database cursor.
        table_name (str): The name of the table to modify.
        column_name (str): The name of the column to add.
        column_type (str): The SQLite data type for the new column (e.g., "TEXT", "INTEGER", "BLOB").
        default_value (Any, optional): A default value for the new column. Only supports
            string, int, or float defaults. Defaults to None.
    """
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col['name'] for col in cursor.fetchall()]
    if column_name not in columns:
        try:
            default_clause = ""
            not_null_clause = ""
            if default_value is not None:
                 if isinstance(default_value, str): default_clause = f"DEFAULT '{default_value}'"
                 elif isinstance(default_value, (int, float)): default_clause = f"DEFAULT {default_value}"
                 else: logger.warning(f"Unsupported default value type for {column_name}: {type(default_value)}")
            # Special case: ensure failure_count is NOT NULL if added this way
            if column_name == "failure_count": not_null_clause = "NOT NULL"

            sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type} {not_null_clause} {default_clause};"
            cursor.execute(sql)
            logger.info(f"Added '{column_name}' column to {table_name} table.")
        except sqlite3.OperationalError as e:
            # Log as warning, as this might happen concurrently or if schema is complex
            logger.warning(f"Could not add '{column_name}' column to {table_name}: {e}")

def initialize_database(db_path: Optional[str] = None):
    """Initializes the database schema, creating or updating the `kb_items` table.

    Ensures the `kb_items` table exists with the required columns and data types.
    Uses `_add_column_if_not_exists` for robustness, allowing the function to
    add missing columns to an existing table (useful for schema evolution).
    Creates necessary indexes on key columns for efficient querying.

    Args:
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.
    """
    effective_path = db_path if db_path is not None else DEFAULT_DB_PATH
    logger.info(f"Initializing database schema in {effective_path}...")
    with get_db_connection(effective_path) as conn: # Use effective_path here
        cursor = conn.cursor()
        # Base table definition with essential columns and constraints
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kb_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_name TEXT UNIQUE NOT NULL,
                item_type TEXT NOT NULL,
                description_nl TEXT,
                lean_code TEXT NOT NULL, /* Initially might contain 'sorry' */
                topic TEXT,
                dependencies TEXT, /* JSON list of unique_names from compilation */
                status TEXT NOT NULL,
                created_at TEXT NOT NULL, /* ISO format string */
                last_modified_at TEXT NOT NULL /* ISO format string */
            );
        """)

        # Add potentially missing columns idempotently
        _add_column_if_not_exists(cursor, "kb_items", "latex_statement", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "latex_proof", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "plan_dependencies", "TEXT") # JSON list
        _add_column_if_not_exists(cursor, "kb_items", "latex_links", "TEXT") # JSON list of dicts
        _add_column_if_not_exists(cursor, "kb_items", "failure_count", "INTEGER", default_value=0)
        _add_column_if_not_exists(cursor, "kb_items", "embedding_latex", "BLOB")
        _add_column_if_not_exists(cursor, "kb_items", "embedding_nl", "BLOB")
        _add_column_if_not_exists(cursor, "kb_items", "latex_review_feedback", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "generation_prompt", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "raw_ai_response", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "lean_error_log", "TEXT")

        # Create indexes for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_unique_name ON kb_items (unique_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_type ON kb_items (item_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_status ON kb_items (status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_topic ON kb_items (topic);")
        conn.commit()
    logger.info("Database schema initialization complete.")


# --- save_kb_item function ---

async def save_kb_item(item: KBItem, client: Optional[GeminiClient] = None, db_path: Optional[str] = None) -> KBItem:
    """Saves a KBItem to the database (INSERT or UPDATE) and handles embedding generation.

    This function performs an UPSERT operation based on the item's `unique_name`.
    If the item is new or if `latex_statement` or `description_nl` has changed
    compared to the existing database record, it attempts to generate new embeddings
    using the provided `GeminiClient` (if available). Embeddings are generated only
    from `latex_statement` (not `latex_proof`) and `description_nl`. Embeddings
    are updated in the database only if they have actually changed or are newly generated.
    The item's `last_modified_at` timestamp is always updated.

    Args:
        item (KBItem): The KBItem object to save. Its `id` attribute will be populated
            or updated after saving.
        client (Optional[GeminiClient]): An initialized GeminiClient instance. Required
            if embedding generation is needed based on content changes. If None,
            embeddings will not be generated or updated.
        db_path (Optional[str]): Path to the database file. If None, uses `DEFAULT_DB_PATH`.

    Returns:
        KBItem: The saved KBItem object, updated with its database `id` and potentially
        newly generated/updated `embedding_nl` and `embedding_latex` fields.

    Raises:
        sqlite3.Error: If a database error occurs during the UPSERT or embedding update.
        TypeError: If the `item.item_type` or `item.status` attributes are not valid
            enum members.
        ValueError: Can be raised by `KBItem.from_db_dict` if fetching the existing
            item fails due to deserialization issues. Can also be raised indirectly
            if embedding generation tasks fail unexpectedly (though most embedding
            errors are caught and logged as warnings).
    """
    if not isinstance(item.item_type, ItemType): raise TypeError(f"item.item_type invalid: {item.item_type}")
    if not isinstance(item.status, ItemStatus): raise TypeError(f"item.status invalid: {item.status}")

    # Ensure latex_proof is None if item type doesn't require it
    if not item.item_type.requires_proof():
        item.latex_proof = None

    # --- Get existing item state from DB (if it exists) to check for changes ---
    existing_item: Optional[KBItem] = None
    effective_db_path = db_path or DEFAULT_DB_PATH
    if item.id is not None:
        existing_item = get_kb_item_by_id(item.id, effective_db_path)
    if existing_item is None and item.unique_name:
         # Try fetching by name if ID fetch failed or ID was None
         existing_item = get_kb_item_by_name(item.unique_name, effective_db_path)

    # --- Determine if text fields relevant for embeddings changed ---
    latex_statement_changed = False # Embeddings based on STATEMENT only
    nl_changed = False
    if existing_item:
        # Compare current item state with DB state
        if item.latex_statement != existing_item.latex_statement:
            latex_statement_changed = True
        if item.description_nl != existing_item.description_nl:
            nl_changed = True
    else: # For new items, consider text changed if it's present
        if item.latex_statement: latex_statement_changed = True
        if item.description_nl: nl_changed = True

    # --- Prepare main data for UPSERT (excluding embeddings initially) ---
    item.last_modified_at = datetime.now(timezone.utc) # Ensure modification time is current
    db_data = item.to_dict_for_db() # Serialize item to DB-compatible dict
    db_data_id = db_data.pop('id', None) # Remove ID if present, use unique_name for conflict
    db_data.pop('embedding_latex', None) # Remove embeddings, handle separately
    db_data.pop('embedding_nl', None)

    columns = ', '.join(db_data.keys())
    placeholders = ', '.join('?' * len(db_data))
    # Update all columns except unique_name on conflict
    update_setters = ', '.join(f"{key} = excluded.{key}" for key in db_data.keys() if key != 'unique_name')

    # SQL for UPSERT using ON CONFLICT...DO UPDATE
    sql_upsert = f"""
        INSERT INTO kb_items ({columns})
        VALUES ({placeholders})
        ON CONFLICT(unique_name) DO UPDATE SET
        {update_setters}
        RETURNING id; -- Return the ID of the inserted or updated row
    """
    params_upsert = tuple(db_data.values())

    # --- Embedding Generation Logic ---
    # Initialize final bytes with current item state (might be None or existing bytes)
    final_latex_bytes = item.embedding_latex
    final_nl_bytes = item.embedding_nl

    # Determine if generation should be attempted
    should_generate_latex = latex_statement_changed and item.latex_statement and client
    should_generate_nl = nl_changed and item.description_nl and client

    kb_search = None # Dynamically import kb_search if needed for generation
    if should_generate_latex or should_generate_nl:
        try:
            # Import necessary embedding generation function dynamically
            from lean_automator.kb import search as kb_search_module
            kb_search = kb_search_module
        except ImportError:
            warnings.warn("kb_search module not found when needed for embedding generation.", ImportWarning)
            kb_search = None # Ensure kb_search is None if import fails

    # Re-evaluate conditions based on successful import
    should_generate_latex = should_generate_latex and kb_search
    should_generate_nl = should_generate_nl and kb_search

    # Issue warnings if generation is needed but cannot proceed
    if (latex_statement_changed and item.latex_statement and not should_generate_latex):
         warnings.warn(f"Cannot generate LaTeX statement embedding for '{item.unique_name}'. Client/kb_search missing or statement empty.")
    if (nl_changed and item.description_nl and not should_generate_nl):
         warnings.warn(f"Cannot generate NL embedding for '{item.unique_name}'. Client/kb_search missing or text empty.")

    # Perform embedding generation if conditions met
    if should_generate_latex or should_generate_nl:
        tasks = []
        # Create task for LaTeX embedding generation or a dummy task
        tasks.append(
             kb_search.generate_embedding(item.latex_statement, EMBEDDING_TASK_TYPE_DOCUMENT, client) # Use latex_statement only
             if should_generate_latex else asyncio.sleep(0, result=None) # Placeholder if no generation needed
        )
        # Create task for NL embedding generation or a dummy task
        tasks.append(
             kb_search.generate_embedding(item.description_nl, EMBEDDING_TASK_TYPE_DOCUMENT, client)
             if should_generate_nl else asyncio.sleep(0, result=None) # Placeholder if no generation needed
        )
        try:
            # Run embedding generation tasks concurrently
            embedding_results = await asyncio.gather(*tasks)
            generated_latex_np = embedding_results[0] # Result from first task
            generated_nl_np = embedding_results[1]    # Result from second task

            # Update final byte values if generation was successful
            if generated_latex_np is not None:
                final_latex_bytes = generated_latex_np.astype(EMBEDDING_DTYPE).tobytes()
                logger.debug(f"Generated LaTeX statement embedding for {item.unique_name}")
            elif should_generate_latex: # Warn if generation was attempted but failed
                 warnings.warn(f"Failed to generate LaTeX statement embedding for '{item.unique_name}'. Using previous value if any.")

            if generated_nl_np is not None:
                final_nl_bytes = generated_nl_np.astype(EMBEDDING_DTYPE).tobytes()
                logger.debug(f"Generated NL embedding for {item.unique_name}")
            elif should_generate_nl: # Warn if generation was attempted but failed
                 warnings.warn(f"Failed to generate NL embedding for '{item.unique_name}'. Using previous value if any.")
        except Exception as e:
             # Catch any broad exception during asyncio.gather or embedding generation
             warnings.warn(f"Error during embedding generation task for '{item.unique_name}': {e}. Using previous values if any.")

    # --- Determine if embedding columns need a separate DB UPDATE ---
    # Compare final bytes (potentially updated by generation) with existing DB bytes
    existing_latex_bytes = existing_item.embedding_latex if existing_item else None
    existing_nl_bytes = existing_item.embedding_nl if existing_item else None
    update_latex_in_db = (final_latex_bytes != existing_latex_bytes)
    update_nl_in_db = (final_nl_bytes != existing_nl_bytes)

    # Update item object's embedding fields AFTER potential generation attempt
    item.embedding_latex = final_latex_bytes
    item.embedding_nl = final_nl_bytes

    # --- Database Operations: UPSERT main data, then UPDATE embeddings if changed ---
    retrieved_id = None
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        try:
            # 1. Execute the UPSERT for main item data, retrieving the ID
            cursor.execute(sql_upsert, params_upsert)
            result = cursor.fetchone()
            if result and result['id'] is not None:
                retrieved_id = result['id']
                item.id = retrieved_id # Update item object with the confirmed database ID
            else:
                # Fallback: If RETURNING didn't work or returned None, try selecting ID
                logger.warning(f"UPSERT RETURNING id failed for {item.unique_name}. Attempting fallback SELECT.")
                cursor.execute("SELECT id FROM kb_items WHERE unique_name = ?", (item.unique_name,))
                result = cursor.fetchone()
                if result and result['id'] is not None:
                    item.id = result['id']
                else:
                    # If ID still not found, something is wrong
                    raise sqlite3.OperationalError(f"Failed to retrieve ID after saving {item.unique_name}")
            logger.debug(f"Upserted item {item.unique_name}, confirmed ID: {item.id}")

            # 2. Conditionally execute UPDATE statements for embeddings if they changed
            if item.id is None:
                 # This should not happen if the UPSERT was successful
                 raise sqlite3.OperationalError(f"Cannot update embeddings, ID is missing for {item.unique_name} after UPSERT.")

            if update_latex_in_db:
                 logger.debug(f"Updating embedding_latex for item ID {item.id}")
                 cursor.execute("UPDATE kb_items SET embedding_latex = ? WHERE id = ?", (final_latex_bytes, item.id))
            if update_nl_in_db:
                 logger.debug(f"Updating embedding_nl for item ID {item.id}")
                 cursor.execute("UPDATE kb_items SET embedding_nl = ? WHERE id = ?", (final_nl_bytes, item.id))

            conn.commit() # Commit transaction after all operations succeed
            logger.info(f"Successfully saved KBItem '{item.unique_name}' (ID: {item.id})")
        except sqlite3.Error as e:
            logger.error(f"Database error during save_kb_item for {item.unique_name}: {e}")
            conn.rollback() # Rollback transaction on error
            raise # Reraise the database error

    return item # Return the item, now guaranteed to have an ID and potentially updated embeddings

# --- Retrieval Functions ---

def get_kb_item_by_id(item_id: int, db_path: Optional[str] = None) -> Optional[KBItem]:
    """Retrieves a single KBItem from the database by its primary key ID.

    Args:
        item_id (int): The integer primary key ID of the item to retrieve.
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.

    Returns:
        Optional[KBItem]: The retrieved KBItem object if found and deserialization
        succeeds, otherwise None. Logs an error if deserialization fails.
    """
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        # Select all columns needed by KBItem.from_db_dict
        cursor.execute("SELECT * FROM kb_items WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        if row:
            try:
                 # Convert sqlite3.Row to dict and deserialize
                 return KBItem.from_db_dict(dict(row))
            except ValueError as e:
                # Log error if deserialization fails for the found row
                logger.error(f"Error deserializing KBItem with id={item_id}: {e}")
                return None
    return None # Return None if no row found with the given ID

def get_kb_item_by_name(unique_name: str, db_path: Optional[str] = None) -> Optional[KBItem]:
    """Retrieves a single KBItem from the database by its unique name.

    Args:
        unique_name (str): The unique string name of the item to retrieve.
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.

    Returns:
        Optional[KBItem]: The retrieved KBItem object if found and deserialization
        succeeds, otherwise None. Logs an error if deserialization fails.
    """
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        # Select all columns needed by KBItem.from_db_dict
        cursor.execute("SELECT * FROM kb_items WHERE unique_name = ?", (unique_name,))
        row = cursor.fetchone()
        if row:
             try:
                 # Convert sqlite3.Row to dict and deserialize
                 return KBItem.from_db_dict(dict(row))
             except ValueError as e:
                 # Log error if deserialization fails for the found row
                 logger.error(f"Error deserializing KBItem with unique_name='{unique_name}': {e}")
                 return None
    return None # Return None if no row found with the given name

def get_items_by_status(status: ItemStatus, db_path: Optional[str] = None) -> Generator[KBItem, None, None]:
    """Retrieves all KBItems matching a specific status, yielding them one by one.

    Args:
        status (ItemStatus): The status enum member to filter by.
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.

    Yields:
        KBItem: KBItem objects matching the specified status. Skips items that
        fail deserialization and logs an error.

    Raises:
        TypeError: If the provided `status` argument is not an `ItemStatus` enum member.
    """
    if not isinstance(status, ItemStatus):
        raise TypeError("Input 'status' must be an ItemStatus enum member.")
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        # Select all columns needed by KBItem.from_db_dict
        cursor.execute("SELECT * FROM kb_items WHERE status = ?", (status.name,)) # Use enum name for query
        for row in cursor:
            try:
                # Yield deserialized item
                yield KBItem.from_db_dict(dict(row))
            except ValueError as e:
                # Log error and continue to next item if deserialization fails
                logger.error(f"Error deserializing KBItem '{row.get('unique_name', 'UNKNOWN')}' while fetching status '{status.name}': {e}")

def get_items_by_topic(topic_prefix: str, db_path: Optional[str] = None) -> Generator[KBItem, None, None]:
    """Retrieves all KBItems whose topic starts with the given prefix, yielding them one by one.

    Performs a case-sensitive prefix search using the SQL LIKE operator.

    Args:
        topic_prefix (str): The prefix string to match against the beginning of the item's topic.
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.

    Yields:
        KBItem: KBItem objects whose topic matches the prefix. Skips items that
        fail deserialization and logs an error.
    """
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        # Select all columns needed by KBItem.from_db_dict
        cursor.execute("SELECT * FROM kb_items WHERE topic LIKE ?", (f"{topic_prefix}%",)) # Use LIKE with % wildcard
        for row in cursor:
            try:
                # Yield deserialized item
                yield KBItem.from_db_dict(dict(row))
            except ValueError as e:
                # Log error and continue to next item if deserialization fails
                logger.error(f"Error deserializing KBItem '{row.get('unique_name', 'UNKNOWN')}' while fetching topic prefix '{topic_prefix}': {e}")

def get_all_items_with_embedding(embedding_field: str, db_path: Optional[str] = None) -> List[Tuple[int, str, bytes]]:
    """Retrieves minimal info (ID, name, embedding) for items with a specific embedding.

    Fetches the primary key ID, unique name, and the raw embedding bytes for all
    items where the specified embedding field (`embedding_nl` or `embedding_latex`)
    is not NULL. This is useful for bulk operations like vector similarity searches.

    Args:
        embedding_field (str): The name of the embedding column to retrieve
            (must be either 'embedding_nl' or 'embedding_latex').
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.

    Returns:
        List[Tuple[int, str, bytes]]: A list of tuples, where each tuple contains
        the item's ID (int), unique name (str), and the embedding data (bytes).
        Returns an empty list if no items have the specified embedding or if a
        database error occurs.

    Raises:
        ValueError: If `embedding_field` is not 'embedding_nl' or 'embedding_latex'.
    """
    if embedding_field not in ['embedding_nl', 'embedding_latex']:
        raise ValueError("embedding_field must be 'embedding_nl' or 'embedding_latex'")
    items_with_embeddings = []
    # Construct SQL query dynamically but safely (field name is validated)
    sql = f"SELECT id, unique_name, {embedding_field} FROM kb_items WHERE {embedding_field} IS NOT NULL;"
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            for row in cursor:
                # Ensure the fetched blob is actually bytes before adding
                blob = row[embedding_field]
                if isinstance(blob, bytes):
                     items_with_embeddings.append((row['id'], row['unique_name'], blob))
                else:
                     # Log a warning if the data type is unexpected (should be BLOB/bytes or NULL)
                     logger.warning(f"Expected bytes for embedding field '{embedding_field}' on item ID {row['id']}, but got type {type(blob)}. Skipping.")
        except sqlite3.Error as e:
            # Log database errors during retrieval
            logger.error(f"Database error retrieving embeddings for field {embedding_field}: {e}")
            # Return empty list on error
            return []
    return items_with_embeddings