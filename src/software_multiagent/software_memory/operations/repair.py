"""Compact repair retrieval, controlled recovery, and episodic compaction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from software_multiagent.software_memory.errors import KnowledgeNotFoundError
from software_multiagent.software_memory.schema.models import (
    CompactEpisode,
    KnowledgeStatus,
    NormalizedFailure,
    OperationContract,
    RepairActionKind,
    RepairCategory,
    RepairDecision,
    RepairEvent,
    RepairKnowledgeItem,
    RepairKnowledgePacket,
    RepairKnowledgeRequest,
    RepairPhase,
    RepairSessionState,
    StatePredicate,
    VerificationState,
    stable_id,
)
from software_multiagent.software_memory.operations.procedural import predicate_slot
from software_multiagent.software_memory.persistence.storage import SQLiteMemoryStore


_CATEGORY_MARKERS = {
    RepairCategory.PROTOCOL: (
        "invalid json",
        "malformed tool",
        "tool call",
        "missing argument",
        "unexpected argument",
    ),
    RepairCategory.ENVIRONMENT: (
        "permission denied",
        "command not found",
        "no such file or directory",
        "connection refused",
        "network",
        "dependency",
        "timeout",
    ),
    RepairCategory.ARTIFACT: (
        "syntax error",
        "parse error",
        "malformed",
        "invalid configuration",
        "while reading",
        "header",
    ),
    RepairCategory.STATE: (
        "precondition",
        "invalid state",
        "conflict",
        "already exists",
        "not initialized",
    ),
    RepairCategory.APPROACH: (
        "unsupported",
        "infeasible",
        "did not converge",
        "diverged",
        "no solution",
    ),
}


def normalize_failure(error_text: str) -> NormalizedFailure:
    text = error_text.casefold()
    text = re.sub(r"(?:[a-z]:[\\/]|/)[^\s:]+", "<path>", text)
    text = re.sub(r"0x[0-9a-f]+", "<hex>", text)
    text = re.sub(
        r"\b(?:line|row|column)\s+\d+\b",
        lambda match: f"{match.group(0).split()[0]} <n>",
        text,
    )
    text = re.sub(r"\b\d+\b", "<n>", text)
    text = " ".join(text.split())[:500]
    category = RepairCategory.UNKNOWN
    for candidate, markers in _CATEGORY_MARKERS.items():
        if any(marker in text for marker in markers):
            category = candidate
            break
    tokens = tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z_][a-z0-9_-]{2,}", text)
            if token not in {"the", "and", "for", "with", "from", "error"}
        )
    )
    return NormalizedFailure(signature=text, category=category, tokens=tokens)


def _matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.casefold() in text.casefold()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z_][a-z0-9_-]{2,}", value.casefold()))


def _estimated_tokens(content: dict[str, Any]) -> int:
    rendered = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, (len(rendered) + 3) // 4)


def _relevant_excerpt(text: str, tokens: tuple[str, ...], limit: int) -> str:
    if len(text) <= limit:
        return text
    lowered = text.casefold()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 4)
    end = min(len(text), start + limit)
    start = max(0, end - limit)
    return text[start:end]


@dataclass
class RepairRetriever:
    store: SQLiteMemoryStore
    evidence_excerpt_characters: int = 800

    def retrieve(self, request: RepairKnowledgeRequest) -> RepairKnowledgePacket:
        if self.store.get_software(request.software_id) is None:
            raise KnowledgeNotFoundError(
                "unknown software identity", software_id=request.software_id
            )
        failure = normalize_failure(request.error_text)
        contracts = self.store.list_contracts(request.software_id)
        workflow_operations: set[str] = set()
        if request.workflow_id:
            workflow = self.store.get_workflow(request.workflow_id)
            if workflow is not None and workflow.software_id == request.software_id:
                workflow_operations.update(
                    step.operation_id
                    for step in workflow.steps
                    if request.workflow_step_id is None
                    or step.step_id == request.workflow_step_id
                )

        successful_episodes = tuple(
            episode
            for episode in self.store.list_episodes(request.software_id)
            if episode.succeeded is True
            and episode.verification == VerificationState.ACCEPTED
            and episode.verifier_id
        )
        current_slots = {predicate_slot(item) for item in request.current_state}
        ranked: list[tuple[float, OperationContract, tuple[str, ...]]] = []
        for contract in contracts:
            if contract.status in {
                KnowledgeStatus.RETRACTED,
                KnowledgeStatus.DEPRECATED,
                KnowledgeStatus.CONFLICTING,
            }:
                continue
            version_score = self._version_score(contract, request.version)
            if version_score == float("-inf"):
                continue
            matched = tuple(
                signature.signature_id
                for signature in contract.failure_signatures
                if _matches(signature.pattern, failure.signature)
            )
            score = version_score
            if contract.contract_id == request.failed_contract_id:
                score += 8.0
            if contract.contract_id in workflow_operations:
                score += 5.0
            score += 12.0 * len(matched)
            overlap = _tokens(
                " ".join(
                    (
                        contract.name,
                        contract.purpose,
                        *contract.inputs,
                        *contract.outputs,
                    )
                )
            ) & set(failure.tokens)
            score += min(4.0, float(len(overlap)))
            state_links = sum(
                predicate_slot(predicate) in current_slots
                for predicate in (*contract.preconditions, *contract.effects)
            )
            score += min(2.0, state_links * 0.5)
            episode_count = sum(
                episode.action_contract_id == contract.contract_id
                for episode in successful_episodes
            )
            score += min(3.0, episode_count * 0.5)
            if score > 0:
                ranked.append((score, contract, matched))
        ranked.sort(key=lambda item: (-item[0], item[1].contract_id))
        for _score, contract, matched in ranked:
            if not matched:
                continue
            signature = next(
                item
                for item in contract.failure_signatures
                if item.signature_id in matched
            )
            failure = failure.model_copy(
                update={"category": RepairCategory(signature.category)}
            )
            break

        proposals: list[RepairKnowledgeItem] = []
        for score, contract, matched in ranked:
            contract_content = {
                "contract_id": contract.contract_id,
                "name": contract.name,
                "interface": contract.interface,
                "inputs": contract.inputs,
                "outputs": contract.outputs,
                "preconditions": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in contract.preconditions
                ],
                "matched_failure_signatures": matched,
                "version_scope": contract.version_scope.model_dump(
                    mode="json", exclude_none=True
                ),
            }
            proposals.append(self._item(contract.contract_id, "contract", score, contract_content))
            for strategy in contract.repair_strategies:
                if matched and strategy.failure_signature_ids and not set(
                    matched
                ).intersection(strategy.failure_signature_ids):
                    continue
                success_count = sum(
                    episode.action_contract_id in strategy.required_operations
                    for episode in successful_episodes
                )
                content = {
                    **strategy.model_dump(mode="json", exclude_none=True),
                    "independently_verified_successes": success_count,
                }
                proposals.append(
                    self._item(
                        strategy.repair_id,
                        "strategy",
                        score + 2.0 + min(3.0, success_count * 0.75),
                        content,
                    )
                )
            for evidence_id in contract.evidence_ids:
                evidence = self.store.get_evidence(evidence_id)
                if evidence is None:
                    continue
                content = {
                    "evidence_id": evidence.evidence_id,
                    "source_uri": evidence.source_uri,
                    "locator": evidence.locator,
                    "source_version": evidence.source_version,
                    "authoritative": evidence.authoritative,
                    "excerpt": _relevant_excerpt(
                        evidence.content,
                        failure.tokens,
                        self.evidence_excerpt_characters,
                    ),
                }
                proposals.append(self._item(evidence_id, "evidence", score - 1.0, content))

        related_ids = {
            entity_id
            for _score, contract, _matched in ranked
            for entity_id in contract.related_entity_ids
        }
        related_ids.update(
            entity_id
            for _score, contract, matched in ranked
            for signature in contract.failure_signatures
            if signature.signature_id in matched
            for entity_id in signature.related_entity_ids
        )
        artifact_names = {value.casefold() for value in request.known_artifacts}
        for entity in self.store.list_entities(request.software_id):
            path = str(entity.attributes.get("path", ""))
            configuration = "config" in entity.entity_type.casefold()
            normalized_path = path.replace("\\", "/").casefold()
            artifact_match = bool(
                normalized_path
                and any(
                    normalized_path == item.replace("\\", "/")
                    or normalized_path.endswith("/" + item.replace("\\", "/"))
                    for item in artifact_names
                )
            )
            token_match = bool(_tokens(f"{entity.name} {path}") & set(failure.tokens))
            if entity.entity_id not in related_ids and not artifact_match and not (
                configuration and token_match
            ):
                continue
            content = {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "name": entity.name,
                "summary": entity.summary,
                "path": path or None,
            }
            proposals.append(self._item(entity.entity_id, "entity", 6.0, content))

        relevant_contracts = {contract.contract_id for _score, contract, _match in ranked}
        for episode in successful_episodes:
            if episode.action_contract_id not in relevant_contracts:
                continue
            proposals.append(
                self._item(
                    episode.episode_id,
                    "episode",
                    4.0,
                    episode.model_dump(mode="json", exclude_none=True),
                )
            )

        proposals.sort(key=lambda item: (-item.score, item.item_type, item.item_id))
        selected: list[RepairKnowledgeItem] = []
        seen: set[tuple[str, str]] = set()
        consumed = 0
        truncated = False
        for item in proposals:
            key = (item.item_type, item.item_id)
            if key in seen:
                continue
            seen.add(key)
            exceeds_items = len(selected) >= request.max_items
            exceeds_tokens = consumed + item.estimated_tokens > request.token_budget
            if exceeds_items or exceeds_tokens:
                truncated = True
                continue
            selected.append(item)
            consumed += item.estimated_tokens
        trace = (
            f"normalized category={failure.category.value} tokens={len(failure.tokens)}",
            f"linked contracts={len(ranked)} workflow_operations={len(workflow_operations)} "
            f"state_predicates={len(current_slots)} artifacts={len(request.known_artifacts)}",
            f"packed items={len(selected)}/{len(proposals)} tokens={consumed}",
        )
        return RepairKnowledgePacket(
            request=request,
            failure=failure,
            items=tuple(selected),
            trace=trace,
            estimated_tokens=consumed,
            truncated=truncated,
        )

    @staticmethod
    def _item(
        item_id: str,
        item_type: str,
        score: float,
        content: dict[str, Any],
    ) -> RepairKnowledgeItem:
        return RepairKnowledgeItem(
            item_id=item_id,
            item_type=item_type,
            score=score,
            content=content,
            estimated_tokens=_estimated_tokens(content),
        )

    @staticmethod
    def _version_score(contract: OperationContract, version: str | None) -> float:
        scope = contract.version_scope
        if not version:
            return -0.5 if scope.exact or scope.compatible else 0.0
        if scope.exact or scope.compatible:
            return 3.0 if scope.matches(version) else float("-inf")
        return 1.0 if scope.matches(version) else float("-inf")


class RepairController:
    """Enforce typed inspect/mutate/rerun/verify recovery transitions."""

    def __init__(self, *, max_inspections: int = 3) -> None:
        if not 2 <= max_inspections <= 3:
            raise ValueError("max_inspections must be two or three")
        self.max_inspections = max_inspections

    def start(self, session_id: str, category: RepairCategory) -> RepairSessionState:
        phase = {
            RepairCategory.ENVIRONMENT: RepairPhase.RERUN,
            RepairCategory.PROTOCOL: RepairPhase.MUTATE,
            RepairCategory.APPROACH: RepairPhase.MUTATE,
        }.get(category, RepairPhase.INSPECT)
        return RepairSessionState(session_id=session_id, category=category, phase=phase)

    def advance(self, state: RepairSessionState, event: RepairEvent) -> RepairDecision:
        if state.phase in {RepairPhase.COMPLETE, RepairPhase.FAILED}:
            return RepairDecision(
                allowed=False,
                state=state,
                reason="repair session is already terminal",
            )
        repeated = self._repeated_without_change(state, event)
        if repeated:
            updated = state.model_copy(
                update={
                    "phase": RepairPhase.MUTATE,
                    "events": (*state.events, event),
                    "last_observed_state": event.state_after,
                    "stop_reason": "retry repeated without observed state change",
                }
            )
            return RepairDecision(
                allowed=False,
                state=updated,
                reason="change the artifact, state, protocol, or approach before rerunning",
                repeated_without_state_change=True,
            )

        allowed, reason = self._allowed(state, event)
        if not allowed:
            return RepairDecision(allowed=False, state=state, reason=reason)
        counts = {
            "inspection_count": state.inspection_count,
            "mutation_count": state.mutation_count,
            "rerun_count": state.rerun_count,
            "verification_count": state.verification_count,
        }
        if event.action == RepairActionKind.INSPECT:
            counts["inspection_count"] += 1
            phase = RepairPhase.INSPECT
        elif event.action in {
            RepairActionKind.MUTATE,
            RepairActionKind.CORRECT_PROTOCOL,
            RepairActionKind.CHANGE_APPROACH,
        }:
            counts["mutation_count"] += 1
            phase = RepairPhase.RERUN
        elif event.action == RepairActionKind.RERUN:
            counts["rerun_count"] += 1
            phase = RepairPhase.VERIFY if event.succeeded else self._failure_phase(state.category)
        else:
            counts["verification_count"] += 1
            phase = (
                RepairPhase.COMPLETE
                if event.succeeded and event.independently_verified
                else self._failure_phase(state.category)
            )
        updated = state.model_copy(
            update={
                **counts,
                "phase": phase,
                "events": (*state.events, event),
                "last_observed_state": event.state_after,
                "stop_reason": None,
            }
        )
        return RepairDecision(allowed=True, state=updated, reason=reason)

    def _allowed(self, state: RepairSessionState, event: RepairEvent) -> tuple[bool, str]:
        if state.phase == RepairPhase.VERIFY and event.action != RepairActionKind.VERIFY:
            return False, "a successful rerun must be independently verified next"
        if event.action == RepairActionKind.INSPECT:
            if state.inspection_count >= self.max_inspections:
                return False, "targeted inspection limit reached"
            return True, "targeted inspection recorded"
        if event.action == RepairActionKind.VERIFY:
            if state.phase != RepairPhase.VERIFY:
                return False, "verification requires a successful rerun"
            if not event.independently_verified:
                return False, "completion requires independent verification"
            return True, "independent verification recorded"
        if state.category in {RepairCategory.ARTIFACT, RepairCategory.STATE}:
            recent = self._after_last_rerun(state)
            if event.action == RepairActionKind.RERUN and not (
                any(item.action == RepairActionKind.INSPECT for item in recent)
                and any(item.action == RepairActionKind.MUTATE for item in recent)
            ):
                return False, (
                    "artifact/state repair requires a new inspection and mutation "
                    "before each rerun"
                )
            if event.action == RepairActionKind.MUTATE and not any(
                item.action == RepairActionKind.INSPECT for item in recent
            ):
                return False, "artifact/state repair requires targeted inspection before mutation"
        if state.category == RepairCategory.PROTOCOL:
            corrected = any(
                item.action == RepairActionKind.CORRECT_PROTOCOL
                for item in self._after_last_rerun(state)
            )
            if event.action == RepairActionKind.RERUN and not corrected:
                return False, "protocol repair requires a corrected structured action"
            if event.action == RepairActionKind.MUTATE:
                return False, "use correct_protocol for protocol failures"
        if state.category == RepairCategory.APPROACH:
            changed = any(
                item.action == RepairActionKind.CHANGE_APPROACH
                for item in self._after_last_rerun(state)
            )
            if event.action == RepairActionKind.RERUN and not changed:
                return False, "approach failure requires a strategy change"
            if event.action == RepairActionKind.MUTATE:
                return False, "use change_approach for approach failures"
        if event.action == RepairActionKind.RERUN:
            return True, "rerun recorded; independent verification remains mandatory"
        if event.action in {
            RepairActionKind.MUTATE,
            RepairActionKind.CORRECT_PROTOCOL,
            RepairActionKind.CHANGE_APPROACH,
        }:
            return True, "repair mutation recorded"
        return False, "action is not valid in the current repair state"

    @staticmethod
    def _failure_phase(category: RepairCategory) -> RepairPhase:
        if category == RepairCategory.ENVIRONMENT:
            return RepairPhase.RERUN
        if category in {RepairCategory.PROTOCOL, RepairCategory.APPROACH}:
            return RepairPhase.MUTATE
        return RepairPhase.INSPECT

    @staticmethod
    def _after_last_rerun(state: RepairSessionState) -> tuple[RepairEvent, ...]:
        last = -1
        for index, item in enumerate(state.events):
            if item.action == RepairActionKind.RERUN:
                last = index
        return state.events[last + 1 :]

    @staticmethod
    def _repeated_without_change(
        state: RepairSessionState,
        event: RepairEvent,
    ) -> bool:
        if event.action != RepairActionKind.RERUN or event.succeeded is not False:
            return False
        previous = next(
            (
                item
                for item in reversed(state.events)
                if item.action == RepairActionKind.RERUN
                and item.operation_id == event.operation_id
            ),
            None,
        )
        return previous is not None and {
            item.key for item in event.state_before
        } == {item.key for item in event.state_after}


def compact_episode(
    observation: Any,
    *,
    software_version: str | None = None,
    failure_signature: str | None = None,
) -> CompactEpisode:
    before = {predicate_slot(item): item for item in observation.state_before}
    changes = tuple(
        item
        for item in observation.state_after
        if before.get(predicate_slot(item)) != item
    )
    outcome = " ".join(str(observation.output_excerpt).split())[:500]
    if not outcome:
        outcome = (
            "succeeded" if observation.succeeded is True else
            "failed" if observation.succeeded is False else "unknown"
        )
    return CompactEpisode(
        episode_id=stable_id("episode", observation.observation_id),
        software_id=observation.software_id,
        observation_id=observation.observation_id,
        task_id=observation.task_id,
        software_version=software_version,
        failure_signature=(
            normalize_failure(failure_signature).signature if failure_signature else None
        ),
        action_contract_id=observation.contract_id,
        state_changes=changes,
        outcome=outcome,
        succeeded=observation.succeeded,
        verification=observation.verification,
        verifier_id=observation.verifier_id,
    )
