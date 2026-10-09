from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _path,
    _schema,
    _SQL_EXECUTE,
    _SQL_LIST,
    _SQL_QUERY,
    _SQL_SCHEMA,
)

def build() -> dict[str, Any]:
    database = _path("Workspace-relative SQLite database path.")
    return {
        "name": "software-bench-sqlite",
        "version": "1.0.0",
        "description": "Inspect and query SQLite databases used by SQL benchmark tasks.",
        "tools": [
            {
                "name": "sqlite_list_tables", "description": "List tables in a SQLite database.",
                "input_schema": _schema({"database": database}, ["database"]),
                "command": ["python", "-c", _SQL_LIST, "{database}"],
            },
            {
                "name": "sqlite_describe_table",
                "description": "Return the CREATE statement for a table.",
                "input_schema": _schema(
                    {"database": database, "table": {"type": "string", "minLength": 1}},
                    ["database", "table"],
                ),
                "command": ["python", "-c", _SQL_SCHEMA, "{database}", "{table}"],
            },
            {
                "name": "sqlite_query",
                "description": "Execute read-only SQL and return at most 500 rows.",
                "input_schema": _schema(
                    {
                        "database": database,
                        "sql": {"type": "string", "minLength": 1, "format": "read-only-sql"},
                    },
                    ["database", "sql"],
                ),
                "command": ["python", "-c", _SQL_QUERY, "{database}", "{sql}"],
            },
            {
                "name": "sqlite_execute",
                "description": "Execute one mutating SQL statement and commit it.",
                "input_schema": _schema(
                    {"database": database, "sql": {"type": "string", "minLength": 1}},
                    ["database", "sql"],
                ),
                "command": ["python", "-c", _SQL_EXECUTE, "{database}", "{sql}"],
                "mutating": True,
            },
            {
                "name": "sqlite_integrity_check",
                "description": "Run SQLite's full integrity check without modifying the database.",
                "input_schema": _schema({"database": database}, ["database"]),
                "command": [
                    "python", "-c",
                    "import sqlite3,sys; c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',"
                    " uri=True); print(*c.execute('PRAGMA integrity_check').fetchone())",
                    "{database}",
                ],
                "timeout_seconds": 300,
            },
        ],
    }
