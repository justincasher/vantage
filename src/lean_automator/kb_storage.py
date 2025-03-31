# File: kb_storage.py

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

# Use absolute imports assuming 'src' is in the path or project is installed
try:
    from lean_automator.llm_call import GeminiClient
except ImportError:
    warnings.warn("llm_call.GeminiClient not found. Embedding generation in save_kb_item will fail.", ImportWarning)
    GeminiClient = None # type: ignore


"""
Defines data structures for the mathematical knowledge base (KBItem, LatexLink, etc.)
and provides functions to interact with an SQLite database for persistent storage.
Uses environment variable KB_DB_PATH for default database location.
Includes storage for embeddings. Compiled Lean object files (.olean) are no longer stored here.
"""

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Sentinel object ---
_sentinel = object()

# --- Database Configuration ---
DEFAULT_DB_PATH = os.getenv('KB_DB_PATH', 'knowledge_base.sqlite')
EMBEDDING_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBEDDING_DTYPE = np.float32

# --- Controlled Vocabularies ---

class ItemType(enum.Enum):
    """Enumerates the different types of items stored in the Knowledge Base."""
    DEFINITION = "Definition"; AXIOM = "Axiom"; THEOREM = "Theorem"; LEMMA = "Lemma"; PROPOSITION = "Proposition"; COROLLARY = "Corollary"; EXAMPLE = "Example"; REMARK = "Remark"; CONJECTURE = "Conjecture"; NOTATION = "Notation"; STRUCTURE = "Structure"

    def requires_proof(self) -> bool:
        """Checks if this item type typically requires a proof."""
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
    """Represents a link to a specific component in an external LaTeX source."""
    citation_text: str
    link_type: str = 'statement'
    source_identifier: Optional[str] = None
    latex_snippet: Optional[str] = None
    match_confidence: Optional[float] = None
    verified_by_human: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LatexLink':
        if 'link_type' not in data: data['link_type'] = 'statement'
        try: return cls(**data)
        except TypeError as e: raise TypeError(f"Error creating LatexLink from dict: {e}") from e

@dataclass
class KBItem:
    """Represents a single node (goal) in the mathematical Knowledge Base."""
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
        """Basic validation after initialization."""
        if not self.unique_name: raise ValueError("unique_name cannot be empty.")
        if not isinstance(self.lean_code, str): self.lean_code = str(self.lean_code)
        if self.created_at.tzinfo is None: self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.last_modified_at.tzinfo is None: self.last_modified_at = self.last_modified_at.replace(tzinfo=timezone.utc)
        # Ensure latex_proof is None if item type doesn't require proof
        if not self.item_type.requires_proof():
            self.latex_proof = None

    def update_status(self, new_status: ItemStatus, error_log: Optional[str] = _sentinel, review_feedback: Optional[str] = _sentinel):
        """Updates status, error log (optional), review feedback (optional), and modification time."""
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
        """Adds a planning dependency if not already present."""
        if dep_unique_name not in self.plan_dependencies:
            self.plan_dependencies.append(dep_unique_name)
            self.last_modified_at = datetime.now(timezone.utc)

    def add_dependency(self, dep_unique_name: str):
        """Adds a discovered Lean dependency if not already present."""
        if dep_unique_name not in self.dependencies:
            self.dependencies.append(dep_unique_name)
            self.last_modified_at = datetime.now(timezone.utc)

    def add_latex_link(self, link: LatexLink):
        """Adds a LaTeX link."""
        self.latex_links.append(link)
        self.last_modified_at = datetime.now(timezone.utc)

    def increment_failure_count(self):
        """Increments the general failure count."""
        self.failure_count += 1
        self.last_modified_at = datetime.now(timezone.utc)

    # update_olean method removed

    # --- Serialization ---
    def to_dict_for_db(self) -> Dict[str, Any]:
        """Serializes to dict for DB, converting complex types."""
        # Ensure proof is None if not required
        proof = self.latex_proof if self.item_type.requires_proof() else None
        # Exclude embeddings here, they are handled separately in save_kb_item
        return {
            "id": self.id,
            "unique_name": self.unique_name,
            "item_type": self.item_type.name,
            "description_nl": self.description_nl,
            "latex_statement": self.latex_statement,
            "latex_proof": proof,
            "lean_code": self.lean_code,
            "topic": self.topic,
            "plan_dependencies": json.dumps(self.plan_dependencies),
            "dependencies": json.dumps(self.dependencies),
            "latex_links": json.dumps([asdict(link) for link in self.latex_links]),
            "status": self.status.name,
            "failure_count": self.failure_count,
            "latex_review_feedback": self.latex_review_feedback,
            "generation_prompt": self.generation_prompt,
            "raw_ai_response": self.raw_ai_response,
            "lean_error_log": self.lean_error_log,
            "created_at": self.created_at.isoformat(),
            "last_modified_at": self.last_modified_at.isoformat()
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'KBItem':
        """Creates KBItem from DB dictionary row."""
        try:
            item_type_str = data["item_type"]
            item_type = ItemType[item_type_str]

            status_str = data.get("status", ItemStatus.PENDING.name)
            try:
                status = ItemStatus[status_str]
            except KeyError:
                warnings.warn(f"Invalid status '{status_str}' for item {data.get('unique_name')}. Defaulting to PENDING.")
                status = ItemStatus.PENDING

            failure_count = data.get("failure_count", 0)
            if not isinstance(failure_count, int):
                warnings.warn(f"Invalid type for failure_count: got {type(failure_count)}. Using 0.")
                failure_count = 0

            # Only load latex_proof if the item type requires it
            latex_proof = data.get("latex_proof") if item_type.requires_proof() else None

            return cls(
                id=data.get("id"),
                unique_name=data["unique_name"],
                item_type=item_type,
                description_nl=data.get("description_nl", ""),
                latex_statement=data.get("latex_statement"),
                latex_proof=latex_proof,
                lean_code=data.get("lean_code", ""),
                embedding_nl=data.get("embedding_nl"),
                embedding_latex=data.get("embedding_latex"),
                topic=data.get("topic", "General"),
                plan_dependencies=json.loads(data.get("plan_dependencies") or '[]'),
                dependencies=json.loads(data.get("dependencies") or '[]'),
                latex_links=[LatexLink.from_dict(link_data) for link_data in json.loads(data.get("latex_links") or '[]')],
                status=status,
                failure_count=failure_count,
                latex_review_feedback=data.get("latex_review_feedback"),
                generation_prompt=data.get("generation_prompt"),
                raw_ai_response=data.get("raw_ai_response"),
                lean_error_log=data.get("lean_error_log"),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_modified_at=datetime.fromisoformat(data["last_modified_at"])
             )
        except (KeyError, json.JSONDecodeError, ValueError, TypeError) as e:
             raise ValueError(f"Error deserializing KBItem '{data.get('unique_name', 'UNKNOWN_ITEM')}' from DB dict: {e}") from e


# --- Database Interaction Functions ---

def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Establishes a connection using specified path or default."""
    effective_path = db_path if db_path is not None else DEFAULT_DB_PATH
    conn = sqlite3.connect(effective_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def _add_column_if_not_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str, column_type: str, default_value: Any = None):
    """Helper to add a column idempotently."""
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col['name'] for col in cursor.fetchall()]
    if column_name not in columns:
        try:
            default_clause = ""
            not_null_clause = ""
            if default_value is not None:
                 if isinstance(default_value, str): default_clause = f"DEFAULT '{default_value}'"
                 elif isinstance(default_value, (int, float)): default_clause = f"DEFAULT {default_value}";
                 else: logger.warning(f"Unsupported default value type for {column_name}: {type(default_value)}")
            # Only force NOT NULL if explicitly required
            if column_name == "failure_count": not_null_clause = "NOT NULL"

            sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type} {not_null_clause} {default_clause};"
            cursor.execute(sql)
            logger.info(f"Added '{column_name}' column to {table_name} table.")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not add '{column_name}' column to {table_name}: {e}")

def initialize_database(db_path: Optional[str] = None):
    """Creates/updates the kb_items table schema and indexes. Assumes starting fresh or columns already exist."""
    effective_path = db_path if db_path is not None else DEFAULT_DB_PATH
    logger.info(f"Initializing database schema in {effective_path}...")
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        # Base table definition with correct, final column names
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kb_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_name TEXT UNIQUE NOT NULL,
                item_type TEXT NOT NULL,
                description_nl TEXT,
                latex_statement TEXT,
                latex_proof TEXT,
                lean_code TEXT NOT NULL,
                embedding_nl BLOB,
                embedding_latex BLOB,
                topic TEXT,
                plan_dependencies TEXT,
                dependencies TEXT,
                latex_links TEXT,
                status TEXT NOT NULL,
                failure_count INTEGER DEFAULT 0 NOT NULL,
                latex_review_feedback TEXT,
                generation_prompt TEXT,
                raw_ai_response TEXT,
                lean_error_log TEXT,
                created_at TEXT NOT NULL,
                last_modified_at TEXT NOT NULL
            );
        """)
        # Use _add_column_if_not_exists for robustness against schema drift or future additions
        # It will do nothing if the column already exists from CREATE TABLE.
        _add_column_if_not_exists(cursor, "kb_items", "latex_statement", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "latex_proof", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "plan_dependencies", "TEXT")
        _add_column_if_not_exists(cursor, "kb_items", "failure_count", "INTEGER", default_value=0)
        _add_column_if_not_exists(cursor, "kb_items", "embedding_latex", "BLOB")
        _add_column_if_not_exists(cursor, "kb_items", "embedding_nl", "BLOB")
        _add_column_if_not_exists(cursor, "kb_items", "latex_review_feedback", "TEXT")

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_unique_name ON kb_items (unique_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_type ON kb_items (item_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_status ON kb_items (status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kbitem_topic ON kb_items (topic);")
        conn.commit()
    logger.info("Database schema initialization complete.")


# --- save_kb_item function ---

async def save_kb_item(item: KBItem, client: Optional[GeminiClient] = None, db_path: Optional[str] = None) -> KBItem:
    """
    Saves (Inserts or Updates) a KBItem, including handling embedding generation
    based ONLY on latex_statement or description_nl if they have changed
    and a client is provided.

    Args:
        item: The KBItem object to save.
        client: An initialized GeminiClient instance (required for embedding generation).
        db_path: Optional path to the database file.

    Returns:
        The saved KBItem object (potentially updated with id and embeddings).

    Raises:
        sqlite3.Error: If a database error occurs.
        TypeError: If item enums are invalid.
        ValueError: If embedding generation fails during an attempt.
    """
    if not isinstance(item.item_type, ItemType): raise TypeError(f"item.item_type invalid: {item.item_type}")
    if not isinstance(item.status, ItemStatus): raise TypeError(f"item.status invalid: {item.status}")

    # Ensure latex_proof is None if item type doesn't require it
    if not item.item_type.requires_proof():
        item.latex_proof = None

    # --- Get existing item state from DB (if it exists) ---
    existing_item: Optional[KBItem] = None
    effective_db_path = db_path or DEFAULT_DB_PATH
    if item.id is not None:
        existing_item = get_kb_item_by_id(item.id, effective_db_path)
    if existing_item is None and item.unique_name:
         existing_item = get_kb_item_by_name(item.unique_name, effective_db_path)

    # --- Determine if text fields relevant for embeddings changed ---
    latex_statement_changed = False # Embeddings based on STATEMENT only
    nl_changed = False
    if existing_item:
        if item.latex_statement != existing_item.latex_statement:
            latex_statement_changed = True
        if item.description_nl != existing_item.description_nl:
            nl_changed = True
    else: # For new items, consider text changed if it's present
        if item.latex_statement: latex_statement_changed = True
        if item.description_nl: nl_changed = True

    # --- Prepare main data for UPSERT (excluding embeddings) ---
    item.last_modified_at = datetime.now(timezone.utc) # Ensure modification time is updated
    db_data = item.to_dict_for_db()
    db_data_id = db_data.pop('id', None)
    db_data.pop('embedding_latex', None) # Handled separately below
    db_data.pop('embedding_nl', None)    # Handled separately below

    columns = ', '.join(db_data.keys())
    placeholders = ', '.join('?' * len(db_data))
    update_setters = ', '.join(f"{key} = excluded.{key}" for key in db_data.keys() if key != 'unique_name')

    sql_upsert = f"""
        INSERT INTO kb_items ({columns})
        VALUES ({placeholders})
        ON CONFLICT(unique_name) DO UPDATE SET
        {update_setters}
        RETURNING id;
    """
    params_upsert = tuple(db_data.values())

    # --- Embedding Logic (based on latex_statement, not latex_proof) ---
    final_latex_bytes = item.embedding_latex
    final_nl_bytes = item.embedding_nl
    should_generate_latex = latex_statement_changed and item.latex_statement and client
    should_generate_nl = nl_changed and item.description_nl and client

    kb_search = None
    if should_generate_latex or should_generate_nl:
        try:
            from lean_automator import kb_search as kb_search_module
            kb_search = kb_search_module
        except ImportError:
            warnings.warn("kb_search module not found when needed for embedding generation.", ImportWarning)
            kb_search = None

    should_generate_latex = should_generate_latex and kb_search
    should_generate_nl = should_generate_nl and kb_search

    if (latex_statement_changed and item.latex_statement and not should_generate_latex):
         warnings.warn(f"Cannot generate LaTeX statement embedding for '{item.unique_name}'. Client/kb_search missing or statement empty.")
    if (nl_changed and item.description_nl and not should_generate_nl):
         warnings.warn(f"Cannot generate NL embedding for '{item.unique_name}'. Client/kb_search missing or text empty.")

    if should_generate_latex or should_generate_nl:
        tasks = []
        tasks.append(
             kb_search.generate_embedding(item.latex_statement, EMBEDDING_TASK_TYPE_DOCUMENT, client) # Use latex_statement
             if should_generate_latex else asyncio.sleep(0, result=None)
        )
        tasks.append(
             kb_search.generate_embedding(item.description_nl, EMBEDDING_TASK_TYPE_DOCUMENT, client)
             if should_generate_nl else asyncio.sleep(0, result=None)
        )
        try:
            embedding_results = await asyncio.gather(*tasks)
            generated_latex_np = embedding_results[0]
            generated_nl_np = embedding_results[1]

            if generated_latex_np is not None:
                final_latex_bytes = generated_latex_np.astype(EMBEDDING_DTYPE).tobytes()
                logger.debug(f"Generated LaTeX statement embedding for {item.unique_name}")
            elif should_generate_latex:
                 warnings.warn(f"Failed to generate LaTeX statement embedding for '{item.unique_name}'. Using previous value if any.")

            if generated_nl_np is not None:
                final_nl_bytes = generated_nl_np.astype(EMBEDDING_DTYPE).tobytes()
                logger.debug(f"Generated NL embedding for {item.unique_name}")
            elif should_generate_nl:
                 warnings.warn(f"Failed to generate NL embedding for '{item.unique_name}'. Using previous value if any.")
        except Exception as e:
             warnings.warn(f"Error during embedding generation task for '{item.unique_name}': {e}. Using previous values if any.")

    # --- Determine if embedding columns need DB update ---
    existing_latex_bytes = existing_item.embedding_latex if existing_item else None
    existing_nl_bytes = existing_item.embedding_nl if existing_item else None
    update_latex_in_db = (final_latex_bytes != existing_latex_bytes)
    update_nl_in_db = (final_nl_bytes != existing_nl_bytes)

    # Update item object state AFTER potential generation attempt
    item.embedding_latex = final_latex_bytes
    item.embedding_nl = final_nl_bytes

    # --- Database Operations ---
    retrieved_id = None
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        try:
            # 1. Upsert main item data using RETURNING id
            cursor.execute(sql_upsert, params_upsert)
            result = cursor.fetchone()
            if result and result['id'] is not None:
                retrieved_id = result['id']
                item.id = retrieved_id # Update item object with confirmed ID
            else:
                # Fallback
                cursor.execute("SELECT id FROM kb_items WHERE unique_name = ?", (item.unique_name,))
                result = cursor.fetchone()
                if result: item.id = result['id']
                else: raise sqlite3.OperationalError(f"Failed to retrieve ID after saving {item.unique_name}")
            logger.debug(f"Upserted item {item.unique_name}, got ID: {item.id}")

            # 2. Conditionally update embeddings using separate UPDATE statements
            if item.id is None:
                 raise sqlite3.OperationalError(f"Cannot update embeddings, ID is missing for {item.unique_name}")

            if update_latex_in_db:
                 logger.debug(f"Updating embedding_latex for item ID {item.id}")
                 cursor.execute("UPDATE kb_items SET embedding_latex = ? WHERE id = ?", (final_latex_bytes, item.id))
            if update_nl_in_db:
                 logger.debug(f"Updating embedding_nl for item ID {item.id}")
                 cursor.execute("UPDATE kb_items SET embedding_nl = ? WHERE id = ?", (final_nl_bytes, item.id))

            conn.commit()
            logger.info(f"Successfully saved KBItem '{item.unique_name}' (ID: {item.id})")
        except sqlite3.Error as e:
            logger.error(f"Database error during save_kb_item for {item.unique_name}: {e}")
            conn.rollback()
            raise

    return item # Return the item (updated with ID and potentially new embeddings)

# --- Retrieval Functions ---

def get_kb_item_by_id(item_id: int, db_path: Optional[str] = None) -> Optional[KBItem]:
    """Retrieves a KBItem by its primary key ID."""
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        # Best practice is to list columns explicitly if schema might vary
        cursor.execute("SELECT * FROM kb_items WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        if row:
            try:
                 return KBItem.from_db_dict(dict(row))
            except ValueError as e:
                logger.error(f"Error deserializing KBItem id={item_id}: {e}")
                return None
    return None

def get_kb_item_by_name(unique_name: str, db_path: Optional[str] = None) -> Optional[KBItem]:
    """Retrieves a KBItem by its unique name."""
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM kb_items WHERE unique_name = ?", (unique_name,))
        row = cursor.fetchone()
        if row:
             try:
                 return KBItem.from_db_dict(dict(row))
             except ValueError as e:
                 logger.error(f"Error deserializing KBItem name='{unique_name}': {e}")
                 return None
    return None

def get_items_by_status(status: ItemStatus, db_path: Optional[str] = None) -> Generator[KBItem, None, None]:
    """Yields KBItems matching a specific status."""
    if not isinstance(status, ItemStatus): raise TypeError("status must be an ItemStatus enum member")
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM kb_items WHERE status = ?", (status.name,))
        for row in cursor:
            try:
                yield KBItem.from_db_dict(dict(row))
            except ValueError as e:
                logger.error(f"Error deserializing KBItem '{row.get('unique_name', 'UNKNOWN')}' fetching status '{status.name}': {e}")

def get_items_by_topic(topic_prefix: str, db_path: Optional[str] = None) -> Generator[KBItem, None, None]:
    """Yields KBItems whose topic starts with the given prefix."""
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM kb_items WHERE topic LIKE ?", (f"{topic_prefix}%",))
        for row in cursor:
            try:
                yield KBItem.from_db_dict(dict(row))
            except ValueError as e:
                logger.error(f"Error deserializing KBItem '{row.get('unique_name', 'UNKNOWN')}' fetching topic '{topic_prefix}': {e}")

def get_all_items_with_embedding(embedding_field: str, db_path: Optional[str] = None) -> List[Tuple[int, str, bytes]]:
    """Retrieves ID, name, and embedding blob for all items having a non-NULL value in the specified embedding field."""
    if embedding_field not in ['embedding_nl', 'embedding_latex']:
        raise ValueError("embedding_field must be 'embedding_nl' or 'embedding_latex'")
    items_with_embeddings = []
    sql = f"SELECT id, unique_name, {embedding_field} FROM kb_items WHERE {embedding_field} IS NOT NULL;"
    effective_db_path = db_path or DEFAULT_DB_PATH
    with get_db_connection(effective_db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            for row in cursor:
                blob = row[embedding_field]
                if isinstance(blob, bytes):
                     items_with_embeddings.append((row['id'], row['unique_name'], blob))
                else:
                     logger.warning(f"Expected bytes for embedding field {embedding_field} on item ID {row['id']}, but got {type(blob)}. Skipping.")
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving embeddings for field {embedding_field}: {e}")
    return items_with_embeddings