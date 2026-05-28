"""
Database session management for the HITL Active Learning Pipeline.

Provides a thread-safe SQLAlchemy session factory and a FastAPI
dependency for request-scoped sessions with automatic rollback on error.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config import settings


# ── Engine Configuration ──────────────────────────────────────
# SQLite-specific: enable WAL mode for concurrent reads during training,
# and enforce foreign key constraints at the database level.
engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record) -> None:
    """
    Enable critical SQLite security and performance pragmas on every connection.

    - journal_mode=WAL: Allows concurrent reads while writing (training runs).
    - foreign_keys=ON: Enforces referential integrity at DB level.
    - secure_delete=ON: Overwrites deleted data to prevent data leakage.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA secure_delete=ON;")
    cursor.close()


# ── Session Factory ───────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ── FastAPI Dependency ────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session for FastAPI dependency injection.

    Automatically commits on success and rolls back on exception,
    ensuring no partial writes corrupt the annotation database.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Standalone Context Manager ────────────────────────────────
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for non-FastAPI usage (scripts, training, CLI).

    Usage:
        with get_db_session() as db:
            tiles = db.query(Tile).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Database Initialization ──────────────────────────────────
def init_db() -> None:
    """
    Create all tables defined by the ORM models.

    Called once at application startup. Safe to call multiple times
    (uses CREATE TABLE IF NOT EXISTS internally).
    """
    from models import Base  # Deferred import to avoid circular dependency

    # Ensure historical dataset model is registered with Base.metadata
    import historical_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
