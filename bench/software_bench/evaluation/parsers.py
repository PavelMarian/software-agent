from __future__ import annotations

import json
import re
from importlib.metadata import entry_points
from typing import Mapping, Protocol


PASSED_STATUSES = {"PASSED", "XFAIL"}
KNOWN_STATUSES = PASSED_STATUSES | {"FAILED", "ERROR", "SKIPPED"}


class LogParser(Protocol):
    def parse(self, output: str) -> Mapping[str, str]: ...


class JsonStatusParser:
    def parse(self, output: str) -> Mapping[str, str]:
        value = json.loads(output.strip())
        if not isinstance(value, dict):
            raise ValueError("JSON test log must be an object")
        result: dict[str, str] = {}
        for test_id, status in value.items():
            if not isinstance(test_id, str) or not isinstance(status, str):
                raise ValueError("JSON test statuses must map strings to strings")
            normalized = status.upper()
            if normalized not in KNOWN_STATUSES:
                raise ValueError(f"unsupported test status: {status}")
            result[test_id] = normalized
        return result


class SimplePytestParser:
    """Small fixture parser; full SWE parsers should be installed as plugins."""

    _prefix = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL)\s+(.+?)\s*$")
    _suffix = re.compile(r"^(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL)\s*$")

    def parse(self, output: str) -> Mapping[str, str]:
        result: dict[str, str] = {}
        for raw_line in output.splitlines():
            line = raw_line.strip()
            match = self._prefix.match(line)
            if match:
                result[match.group(2)] = match.group(1)
                continue
            match = self._suffix.match(line)
            if match:
                result[match.group(1)] = match.group(2)
        return result


def load_log_parser(parser_id: str) -> LogParser:
    if parser_id == "json-status":
        return JsonStatusParser()
    if parser_id in {"pytest", "parse_log_pytest", "parse_log_pytest_v2"}:
        return SimplePytestParser()
    matches = entry_points(group="software_bench.log_parsers", name=parser_id)
    if not matches:
        raise LookupError(
            f"log parser {parser_id!r} is not installed; expected an entry point in "
            "the 'software_bench.log_parsers' group"
        )
    return matches[0].load()()
