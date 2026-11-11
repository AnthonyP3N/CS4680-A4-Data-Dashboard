"""Creates and tracks the data sources available to one user session.

A session starts with a private copy of the sample store. Uploaded files go
into their own data source ("Uploaded files"), never into the sample, so each
source keeps its own meaning, examples and conversation. One visitor's data
sources are never visible to another's.

New kinds of data source plug in through `register_adapter_type`.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from storage.models import DataSourceRecord

from . import sample_store
from .adapters import DataSourceAdapter, DataSourceError, DataSourceLimits
from .metadata import SemanticConfig
from .seed import build_database
from .sqlite_adapter import SQLiteAdapter

if TYPE_CHECKING:
    from storage import Storage

UPLOADS_NAME = "Uploaded files"
UPLOADS_DESCRIPTION = "Tables created from the files you uploaded."

_master_lock = threading.Lock()
_master: dict[str, Path] = {}      # day -> path of the shared read-only master copy

# type name -> factory(**settings) returning an adapter. Lets a deployment add
# e.g. a PostgreSQL adapter without touching the agent or the UI.
ADAPTER_TYPES: dict[str, Callable[..., DataSourceAdapter]] = {"sqlite": SQLiteAdapter}


def register_adapter_type(type_name: str, factory: Callable[..., DataSourceAdapter]) -> None:
    ADAPTER_TYPES[type_name.lower()] = factory


def create_adapter(type_name: str, **settings) -> DataSourceAdapter:
    """Build an adapter of a registered type. `settings` are passed to its factory and are not stored here."""
    try:
        factory = ADAPTER_TYPES[type_name.lower()]
    except KeyError:
        known = ", ".join(sorted(ADAPTER_TYPES))
        raise DataSourceError(f"No adapter is installed for '{type_name}'. Available: {known}.") from None
    return factory(**settings)


def master_sample_db(directory: Path, builder: Callable[[Path], Path] = build_database) -> Path:
    """Build the master sample database once per day (its dates are relative to today)."""
    day = date.today().isoformat()
    with _master_lock:
        path = _master.get(day)
        if path is None or not path.exists():
            path = builder(directory / "sample.db")
            _master.clear()
            _master[day] = path
        return path


def to_record(source: DataSourceAdapter, workspace_id: str) -> DataSourceRecord:
    """The storable description of a data source: metadata and business context, never credentials."""
    info = source.info
    return DataSourceRecord(
        id=info.id, workspace_id=workspace_id, name=info.name, source_type=info.type, dialect=info.dialect,
        description=info.description, connection=dict(info.connection), writable=info.writable,
        is_sample=info.is_sample, semantic_config=source.semantics.to_dict(),
        last_schema_refresh=info.last_schema_refresh.isoformat(timespec="seconds") if info.last_schema_refresh else None,
        created_at=info.created_at.isoformat(timespec="milliseconds"))


class DataSourceRegistry:
    """The data sources of one workspace, with one of them active.

    With `storage` and `workspace_id` set, the list of sources and each one's
    business context are saved, and `load()` brings them back after a restart.
    The data files live in `work_dir`, which belongs to this workspace alone.
    """

    def __init__(self, sample_dir: Path | None = None, work_dir: Path | None = None, *,
                 storage: "Storage | None" = None, workspace_id: str | None = None):
        self.sample_dir = sample_dir or Path(__file__).parent
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="askdata_"))
        self._owns_work_dir = work_dir is None
        self.sources: dict[str, DataSourceAdapter] = {}
        self.active_id: str | None = None
        self.storage = storage
        self.workspace_id = workspace_id
        self.limits: DataSourceLimits | None = None   # when set, every source in this registry uses these limits

    # -- persistence ------------------------------------------------------- #
    def attach(self, storage: "Storage", workspace_id: str) -> None:
        """Start saving this registry's sources to `storage` under `workspace_id`."""
        self.storage, self.workspace_id = storage, workspace_id
        for source in self.sources.values():
            self.persist(source)

    def persist(self, source: DataSourceAdapter) -> None:
        if self.storage is not None and self.workspace_id is not None:
            self.storage.data_sources.save(to_record(source, self.workspace_id))

    def load(self) -> None:
        """Rebuild the sources recorded for this workspace; create the sample if there are none."""
        records = self.storage.data_sources.list(self.workspace_id) if self.storage and self.workspace_id else []
        today = date.today().isoformat()
        for record in records:
            file_name = str(record.connection.get("file", ""))
            path = self.work_dir / file_name
            if not file_name or Path(file_name).name != file_name:      # never follow a path out of work_dir
                continue
            semantics = SemanticConfig.from_dict(record.semantic_config) if record.semantic_config else None
            if record.is_sample:
                # Sample dates are relative to today, so yesterday's copy is replaced (edits to it are lost)
                if not path.exists() or record.connection.get("built_on") != today:
                    self.work_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(master_sample_db(self.sample_dir), path)
            elif not path.exists():
                continue                                                # its conversations stay stored but hidden
            source = SQLiteAdapter(path, name=record.name, description=record.description,
                                   writable=record.writable, is_sample=record.is_sample, source_id=record.id,
                                   source_type=record.source_type, semantics=semantics)
            if record.is_sample:
                source.info.connection["built_on"] = today
            if self.limits is not None:
                source.info.limits = self.limits
            self.sources[source.info.id] = source
            self.persist(source)
        if self.sample is None:
            self.create_sample()
        if self.active_id is None:
            self.active_id = self.sample.info.id

    # -- access ------------------------------------------------------------ #
    @property
    def active(self) -> DataSourceAdapter:
        if self.active_id is None or self.active_id not in self.sources:
            raise DataSourceError("No data source is selected.")
        return self.sources[self.active_id]

    def list(self) -> list[DataSourceAdapter]:
        return list(self.sources.values())

    def get(self, source_id: str) -> DataSourceAdapter:
        try:
            return self.sources[source_id]
        except KeyError:
            raise DataSourceError("That data source doesn't exist in this session.") from None

    def add(self, source: DataSourceAdapter, *, activate: bool = True) -> DataSourceAdapter:
        if self.limits is not None:
            source.info.limits = self.limits
        self.sources[source.info.id] = source
        if activate or self.active_id is None:
            self.active_id = source.info.id
        self.persist(source)
        return source

    def select(self, source_id: str) -> DataSourceAdapter:
        source = self.get(source_id)       # raises for ids that aren't this session's
        self.active_id = source_id
        return source

    def remove(self, source_id: str) -> None:
        source = self.sources.pop(source_id, None)
        if source is None:
            return
        if self.storage is not None and self.workspace_id is not None:
            self.storage.data_sources.delete(self.workspace_id, source_id)   # its conversations go with it
        path = getattr(source, "path", None)
        if isinstance(path, Path) and self.work_dir in path.parents:
            path.unlink(missing_ok=True)   # only ever deletes files this session created
        if self.active_id == source_id:
            self.active_id = next(iter(self.sources), None)

    # -- sample data ------------------------------------------------------- #
    @property
    def sample(self) -> SQLiteAdapter | None:
        return next((s for s in self.sources.values() if s.info.is_sample and isinstance(s, SQLiteAdapter)), None)

    def create_sample(self) -> SQLiteAdapter:
        """Add a private, writable copy of the sample store and make it active."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        dst = self.work_dir / "store.db"
        shutil.copy(master_sample_db(self.sample_dir), dst)
        source = SQLiteAdapter(dst, name=sample_store.NAME, description=sample_store.DESCRIPTION,
                               writable=True, is_sample=True, semantics=sample_store.config())
        source.info.connection["built_on"] = date.today().isoformat()
        return self.add(source)

    def reset_sample(self, source: SQLiteAdapter) -> None:
        """Throw away edits by restoring the original sample data and its configuration."""
        if not source.info.is_sample:
            raise DataSourceError("Only the sample data can be reset.")
        shutil.copy(master_sample_db(self.sample_dir), source.path)
        source.semantics = sample_store.config()
        source.schema(refresh=True)
        source.info.connection["built_on"] = date.today().isoformat()
        self.persist(source)

    # -- uploads ----------------------------------------------------------- #
    @property
    def uploads(self) -> SQLiteAdapter | None:
        return next((s for s in self.sources.values() if s.info.type == "csv" and isinstance(s, SQLiteAdapter)), None)

    def ensure_uploads(self) -> SQLiteAdapter:
        """The data source that holds uploaded files, created empty the first time it is needed."""
        existing = self.uploads
        if existing is not None:
            return existing
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path = self.work_dir / "uploads.db"
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA user_version = 1")   # forces the file to be created
            conn.commit()
        finally:
            conn.close()
        return self.add(SQLiteAdapter(path, name=UPLOADS_NAME, description=UPLOADS_DESCRIPTION,
                                      writable=True, source_type="csv"), activate=False)

    def cleanup(self) -> None:
        """Delete this session's private files."""
        if self._owns_work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)
        self.sources.clear()
        self.active_id = None
