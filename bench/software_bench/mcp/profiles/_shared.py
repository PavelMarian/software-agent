from __future__ import annotations

from typing import Any, Mapping


_SQL_LIST = (
    "import json,sqlite3,sys; "
    "c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro', uri=True); "
    "print(json.dumps([r[0] for r in c.execute(\"SELECT name FROM sqlite_master "
    "WHERE type='table' ORDER BY name\")]))"
)
_SQL_SCHEMA = (
    "import json,sqlite3,sys; "
    "c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro', uri=True); "
    "print(json.dumps(list(c.execute(\"SELECT sql FROM sqlite_master WHERE name=?\", "
    "(sys.argv[2],))), default=str))"
)
_SQL_QUERY = (
    "import json,sqlite3,sys; "
    "c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro', uri=True); "
    "c.execute('PRAGMA query_only=ON'); "
    "cur=c.execute(sys.argv[2]); cols=[d[0] for d in cur.description or []]; "
    "print(json.dumps({'columns':cols,'rows':cur.fetchmany(500)}, default=str))"
)
_SQL_EXECUTE = (
    "import json,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
    "cur=c.execute(sys.argv[2]); c.commit(); "
    "print(json.dumps({'rows_changed':cur.rowcount}))"
)


def _schema(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required,
        "additionalProperties": False,
    }


def _path(description: str) -> dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "format": "workspace-path",
        "description": description,
    }


def _arguments(
    description: str = "Additional argv items passed without a shell.",
) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": description,
    }
