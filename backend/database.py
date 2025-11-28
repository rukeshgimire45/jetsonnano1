from __future__ import annotations

from contextlib import contextmanager
from sqlmodel import SQLModel, create_engine, Session

from .config import config

engine = create_engine(f"sqlite:///{config.db_path}", echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope() -> Session:
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
