from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Any


@dataclass
class GeminiConnectedEvent(PluginBaseEvent):
    """Event emitted when Gemini realtime connection is established."""

    type: str = field(default="plugin.gemini.connected", init=False)
    model: str | None = None


@dataclass
class GeminiErrorEvent(PluginBaseEvent):
    """Event emitted when Gemini encounters an error."""

    type: str = field(default="plugin.gemini.error", init=False)
    error: Any | None = None


@dataclass
class GeminiAudioEvent(PluginBaseEvent):
    """Event emitted when Gemini provides audio output."""

    type: str = field(default="plugin.gemini.audio", init=False)
    audio_data: bytes | None = None


@dataclass
class GeminiTextEvent(PluginBaseEvent):
    """Event emitted when Gemini provides text output."""

    type: str = field(default="plugin.gemini.text", init=False)
    text: str | None = None


@dataclass
class GeminiResponseEvent(PluginBaseEvent):
    """Event emitted when Gemini provides a response chunk."""

    type: str = field(default="plugin.gemini.response", init=False)
    response_chunk: Any | None = None
