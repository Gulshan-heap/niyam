import asyncio
import json

import pytest
from sqlalchemy.exc import OperationalError

from niyam.db.session import get_engine
from niyam.mcp_server import server


def call(name: str, args: dict) -> dict:
    result = asyncio.run(server.call_tool(name, args))
    return json.loads(result.content[0].text)


def test_tools_are_registered():
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {
        "search_regulations",
        "get_circular",
        "get_amendment_chain",
        "ask",
    }


def test_unknown_document_is_an_error_not_a_crash():
    try:
        get_engine().connect().close()
    except OperationalError:
        pytest.skip("Postgres not reachable")
    assert "error" in call("get_circular", {"source_id": "0"})
    assert "error" in call("get_amendment_chain", {"source_id": "0"})
