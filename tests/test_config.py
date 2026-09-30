from __future__ import annotations

from software_multiagent.config import load_environment, target_model


def test_load_environment_preserves_process_values(tmp_path, monkeypatch) -> None:
    environment_file = tmp_path / ".env"
    environment_file.write_text(
        "SOFTWARE_MULTIAGENT_PROVIDER=openrouter\n"
        "SOFTWARE_MULTIAGENT_MODEL='openai/gpt-4.1'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOFTWARE_MULTIAGENT_PROVIDER", "openai")
    monkeypatch.delenv("SOFTWARE_MULTIAGENT_MODEL", raising=False)

    assert load_environment(environment_file) == environment_file.resolve()
    assert target_model() == ("openai", "openai/gpt-4.1")


def test_load_environment_can_override_process_values(tmp_path, monkeypatch) -> None:
    environment_file = tmp_path / ".env"
    environment_file.write_text(
        "SOFTWARE_MULTIAGENT_PROVIDER=openrouter\n"
        "SOFTWARE_MULTIAGENT_MODEL=openai/gpt-4.1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOFTWARE_MULTIAGENT_PROVIDER", "openai")

    load_environment(environment_file, override=True)

    assert target_model() == ("openrouter", "openai/gpt-4.1")
