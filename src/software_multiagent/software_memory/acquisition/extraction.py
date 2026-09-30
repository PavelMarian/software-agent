from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from software_multiagent.software_memory.acquisition.models import (
    ApiOperation,
    ApiParameter,
    ApiResponse,
    EvidenceRef,
    KnowledgeStatus,
    SourceDocument,
    SourceSection,
    stable_id,
)


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}


class JsonExtractionClient(Protocol):
    """Minimal model port; adapters may use LangChain, an SDK, or a local model."""

    def extract_json(
        self, *, system_prompt: str, document: str, json_schema: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class _ExtractedOperations(BaseModel):
    operations: list[ApiOperation]


def _pointer_evidence(document: SourceDocument, pointer: str, value: Any) -> EvidenceRef:
    quote = json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:2_000]
    section_id = document.sections[0].section_id if document.sections else None
    return EvidenceRef(
        source_id=document.source_id,
        section_id=section_id,
        source_url=document.canonical_url,
        quote=quote,
        pointer=pointer,
        verified=True,
    )


def _resolve_local_ref(root: Mapping[str, Any], value: Any) -> Any:
    if not isinstance(value, Mapping) or "$ref" not in value:
        return value
    ref = value.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return value
    current: Any = root
    for segment in ref[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or segment not in current:
            return value
        current = current[segment]
    return current


class OpenApiExtractor:
    """Loss-minimizing conversion of OpenAPI/Swagger JSON into operation contracts."""

    def can_extract(self, document: SourceDocument) -> bool:
        try:
            data = json.loads(document.raw_text)
        except json.JSONDecodeError:
            return False
        return isinstance(data, Mapping) and isinstance(data.get("paths"), Mapping)

    def extract(self, document: SourceDocument) -> tuple[ApiOperation, ...]:
        data = json.loads(document.raw_text)
        if not isinstance(data, Mapping) or not isinstance(data.get("paths"), Mapping):
            return ()
        base_urls = self._base_urls(data)
        global_security = data.get("security", [])
        operations: list[ApiOperation] = []
        for path, path_item_raw in data["paths"].items():
            path_item = _resolve_local_ref(data, path_item_raw)
            if not isinstance(path_item, Mapping):
                continue
            shared_parameters = path_item.get("parameters", [])
            for method, operation_raw in path_item.items():
                if method.lower() not in HTTP_METHODS:
                    continue
                operation = _resolve_local_ref(data, operation_raw)
                if not isinstance(operation, Mapping):
                    continue
                pointer = f"#/paths/{str(path).replace('~', '~0').replace('/', '~1')}/{method}"
                evidence = (_pointer_evidence(document, pointer, operation),)
                parameters = self._parameters(
                    data,
                    [*shared_parameters, *operation.get("parameters", [])],
                    document,
                    f"{pointer}/parameters",
                )
                request_schema, request_types = self._request_body(data, operation.get("requestBody"))
                responses = self._responses(data, operation.get("responses", {}), document, pointer)
                raw_name = operation.get("operationId") or operation.get("summary") or f"{method}_{path}"
                operation_id = stable_id("op", method.upper(), str(path))
                security = operation.get("security", global_security)
                auth = tuple(
                    dict.fromkeys(
                        name
                        for requirement in security if isinstance(requirement, Mapping)
                        for name in requirement
                    )
                )
                operations.append(
                    ApiOperation(
                        operation_id=operation_id,
                        name=str(raw_name),
                        method=method,
                        path=str(path),
                        base_urls=base_urls,
                        description=operation.get("description") or operation.get("summary"),
                        parameters=parameters,
                        request_body_schema=request_schema,
                        request_content_types=request_types,
                        responses=responses,
                        authentication=auth,
                        tags=tuple(str(tag) for tag in operation.get("tags", [])),
                        deprecated=bool(operation.get("deprecated", False)),
                        status=(
                            KnowledgeStatus.DEPRECATED
                            if operation.get("deprecated")
                            else KnowledgeStatus.EVIDENCE_SUPPORTED
                        ),
                        evidence=evidence,
                    )
                )
        return tuple(operations)

    @staticmethod
    def _base_urls(data: Mapping[str, Any]) -> tuple[str, ...]:
        servers = data.get("servers", [])
        urls = [server.get("url") for server in servers if isinstance(server, Mapping)]
        if not urls and data.get("host"):
            schemes = data.get("schemes") or ["https"]
            base_path = data.get("basePath", "")
            urls = [f"{scheme}://{data['host']}{base_path}" for scheme in schemes]
        return tuple(str(url) for url in urls if url)

    @staticmethod
    def _schema_type(schema: Any) -> str | None:
        if not isinstance(schema, Mapping):
            return None
        if schema.get("type"):
            return str(schema["type"])
        if "$ref" in schema:
            return str(schema["$ref"]).rsplit("/", 1)[-1]
        return None

    def _parameters(
        self,
        root: Mapping[str, Any],
        values: list[Any],
        document: SourceDocument,
        pointer: str,
    ) -> tuple[ApiParameter, ...]:
        parameters: list[ApiParameter] = []
        for index, raw in enumerate(values):
            parameter = _resolve_local_ref(root, raw)
            if not isinstance(parameter, Mapping) or not parameter.get("name"):
                continue
            schema = _resolve_local_ref(root, parameter.get("schema", {}))
            schema = schema if isinstance(schema, Mapping) else {}
            evidence = (_pointer_evidence(document, f"{pointer}/{index}", parameter),)
            constraints = {
                key: deepcopy(schema[key])
                for key in ("minimum", "maximum", "minLength", "maxLength", "pattern", "format")
                if key in schema
            }
            raw_location = str(parameter.get("in", "unknown"))
            location = raw_location if raw_location in {"path", "query", "header", "cookie", "body"} else "unknown"
            parameters.append(
                ApiParameter(
                    name=str(parameter["name"]),
                    location=location,
                    required=bool(parameter.get("required", False) or parameter.get("in") == "path"),
                    schema_type=self._schema_type(schema),
                    description=parameter.get("description"),
                    default=schema.get("default", parameter.get("default")),
                    example=parameter.get("example", schema.get("example")),
                    enum=tuple(schema.get("enum", ())),
                    constraints=constraints,
                    evidence=evidence,
                )
            )
        return tuple(parameters)

    @staticmethod
    def _request_body(root: Mapping[str, Any], raw: Any) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
        body = _resolve_local_ref(root, raw)
        if not isinstance(body, Mapping):
            return None, ()
        content = body.get("content", {})
        if not isinstance(content, Mapping):
            return None, ()
        schemas: dict[str, Any] = {}
        for media_type, media in content.items():
            if isinstance(media, Mapping) and "schema" in media:
                schemas[str(media_type)] = deepcopy(media["schema"])
        if len(schemas) == 1:
            return next(iter(schemas.values())), tuple(schemas)
        return (schemas or None), tuple(str(media_type) for media_type in content)

    @staticmethod
    def _responses(
        root: Mapping[str, Any], raw: Any, document: SourceDocument, pointer: str
    ) -> tuple[ApiResponse, ...]:
        if not isinstance(raw, Mapping):
            return ()
        responses: list[ApiResponse] = []
        for status, raw_response in raw.items():
            response = _resolve_local_ref(root, raw_response)
            if not isinstance(response, Mapping):
                continue
            content = response.get("content", {})
            content = content if isinstance(content, Mapping) else {}
            schema: dict[str, Any] | None = None
            example: Any = None
            if content:
                first = next(iter(content.values()))
                if isinstance(first, Mapping):
                    schema = deepcopy(first.get("schema"))
                    example = first.get("example")
            responses.append(
                ApiResponse(
                    status_code=str(status),
                    description=response.get("description"),
                    content_types=tuple(str(item) for item in content),
                    response_schema=schema,
                    example=example,
                    evidence=(_pointer_evidence(document, f"{pointer}/responses/{status}", response),),
                )
            )
        return tuple(responses)


class Doc2AgentExtractor:
    """Direct extraction plus compact endpoint fingerprints, inspired by Doc2Agent.

    OpenAPI documents are converted deterministically. Unstructured HTML/Markdown can be
    processed by a caller-supplied structured-output model. Without a model, conservative
    method/path signatures are extracted and no undocumented values are invented.
    """

    def __init__(self, client: JsonExtractionClient | None = None) -> None:
        self.client = client
        self.openapi = OpenApiExtractor()

    def extract(self, document: SourceDocument) -> tuple[ApiOperation, ...]:
        if self.openapi.can_extract(document):
            return self.openapi.extract(document)
        if self.client is not None:
            return self._model_extract(document)
        return self._fingerprint_extract(document)

    def _model_extract(self, document: SourceDocument) -> tuple[ApiOperation, ...]:
        schema = _ExtractedOperations.model_json_schema()
        section_text = "\n\n".join(
            f"[section_id={section.section_id}]\n{section.content}"
            for section in document.sections
        )
        result = self.client.extract_json(
            system_prompt=self.system_prompt(document), document=section_text, json_schema=schema
        )
        try:
            operations = _ExtractedOperations.model_validate(result).operations
        except ValidationError as error:
            raise ValueError(f"structured API extraction did not match the contract: {error}") from error
        return tuple(self._verify_evidence(document, operation) for operation in operations)

    @staticmethod
    def system_prompt(document: SourceDocument) -> str:
        return (
            "Extract only HTTP API operations explicitly supported by the supplied documentation. "
            "Return JSON matching the supplied schema. Preserve requiredness, location, types, "
            "defaults, enums, constraints, request bodies, responses, authentication and deprecation. "
            "Do not invent missing URLs, parameter values, defaults, examples, or response schemas. "
            "For every operation and parameter, attach EvidenceRef with the exact supplied section_id, "
            "source_id='" + document.source_id + "', source_url='" + document.canonical_url + "', and "
            "a short verbatim quote. Use stable operation_id based on METHOD and path when possible."
        )

    @staticmethod
    def _verify_evidence(document: SourceDocument, operation: ApiOperation) -> ApiOperation:
        sections = {section.section_id: section for section in document.sections}

        def checked(reference: EvidenceRef) -> EvidenceRef:
            section = sections.get(reference.section_id or "")
            valid = bool(section and reference.quote and reference.quote in section.content)
            return reference.model_copy(
                update={
                    "source_id": document.source_id,
                    "source_url": document.canonical_url,
                    "verified": valid,
                }
            )

        params = tuple(
            parameter.model_copy(update={"evidence": tuple(checked(ref) for ref in parameter.evidence)})
            for parameter in operation.parameters
        )
        evidence = tuple(checked(ref) for ref in operation.evidence)
        supported = bool(evidence) and all(ref.verified for ref in evidence)
        return operation.model_copy(
            update={
                "operation_id": stable_id("op", operation.method, operation.path),
                "parameters": params,
                "evidence": evidence,
                "status": KnowledgeStatus.EVIDENCE_SUPPORTED if supported else KnowledgeStatus.EXTRACTED,
            }
        )

    @staticmethod
    def _fingerprint_extract(document: SourceDocument) -> tuple[ApiOperation, ...]:
        operations: list[ApiOperation] = []
        seen: set[tuple[str, str]] = set()
        pattern = re.compile(
            r"(?im)(?:^|[`|\s])(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+"
            r"((?:https?://[^\s`|]+)|(?:/[^\s`|]+))"
        )
        for section in document.sections:
            for match in pattern.finditer(section.content):
                method, path = match.group(1).upper(), match.group(2).rstrip(".,;)")
                if path.startswith(("http://", "https://")):
                    parsed = urlparse(path)
                    base_urls = (f"{parsed.scheme}://{parsed.netloc}",)
                    path = parsed.path or "/"
                    if parsed.query:
                        path += f"?{parsed.query}"
                else:
                    base_urls = ()
                key = (method, path)
                if key in seen:
                    continue
                seen.add(key)
                quote = match.group(0).strip()
                evidence = EvidenceRef(
                    source_id=document.source_id,
                    section_id=section.section_id,
                    source_url=document.canonical_url,
                    heading_path=section.heading_path,
                    quote=quote,
                    verified=True,
                )
                operations.append(
                    ApiOperation(
                        operation_id=stable_id("op", method, path),
                        name=f"{method} {path}",
                        method=method,
                        path=path,
                        base_urls=base_urls,
                        status=KnowledgeStatus.EVIDENCE_SUPPORTED,
                        evidence=(evidence,),
                    )
                )
        return tuple(operations)
