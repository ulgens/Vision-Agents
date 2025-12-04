from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Any


@dataclass
class OpenAIStreamEvent(PluginBaseEvent):
    """Event emitted when OpenAI provides a stream event."""

    type: str = field(default="plugin.openai.stream", init=False)
    event_type: str | None = None
    event_data: Any | None = None


@dataclass
class LLMErrorEvent(PluginBaseEvent):
    """Event emitted when an LLM encounters an error."""

    type: str = field(default="plugin.llm.error", init=False)
    error_message: str | None = None
    event_data: Any | None = None
