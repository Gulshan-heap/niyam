from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from niyam.config import get_settings


@lru_cache
def get_engine() -> Engine:
    # Keepalives stop NATs and serverless Postgres hosts from dropping long-lived connections.
    keepalives = {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
    return create_engine(get_settings().database_url, pool_pre_ping=True, connect_args=keepalives)


def get_session() -> Iterator[Session]:
    with sessionmaker(bind=get_engine())() as session:
        yield session
