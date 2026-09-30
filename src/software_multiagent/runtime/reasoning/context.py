"""Deterministic bounded working context; the full execution journal is retained."""
from dataclasses import replace
from software_multiagent.core.contracts import Message


class BoundedContext:
    def build(self, messages, agent_id, knowledge, max_characters, *, working_state=""):
        if max_characters <= 0:
            raise ValueError("context budget must be positive")
        # Preserve current execution state separately from stale conversation.
        pinned = working_state[:max_characters // 4]
        pack = knowledge[:min(6000, max_characters // 3)]
        remaining = max_characters - len(pack) - len(pinned)
        visible = []
        for message in reversed(messages):
            if message.recipient not in (None, agent_id):
                continue
            if len(message.content) > remaining:
                prefix = (f"Observation {message.metadata['observation_id']} "
                          f"from {message.sender}: [truncated] "
                          if message.kind == "observation" else "[truncated] ")
                tail = max(0, remaining - len(prefix))
                clipped = prefix[:remaining] + (message.content[-tail:] if tail else "")
                visible.append(replace(message, content=clipped))
                break
            visible.append(message)
            remaining -= len(message.content)
            if remaining <= 0:
                break
        visible.reverse()
        if pack:
            visible.insert(0, Message("knowledge", pack, agent_id, "context"))
        if pinned:
            visible.insert(0, Message("runtime", pinned, agent_id, "working_state"))
        return tuple(visible)
