from dataclasses import dataclass, field

from getstream.video.rtc import PcmData

from vision_agents.core.events import PluginBaseEvent
from typing import Any
import uuid


@dataclass
class RealtimeConnectedEvent(PluginBaseEvent):
    """Event emitted when realtime connection is established."""

    type: str = field(default="plugin.realtime_connected", init=False)
    provider: str | None = None
    session_config: dict[str, Any] | None = None
    capabilities: list[str] | None = None


@dataclass
class RealtimeDisconnectedEvent(PluginBaseEvent):
    type: str = field(default="plugin.realtime_disconnected", init=False)
    provider: str | None = None
    reason: str | None = None
    was_clean: bool = True


@dataclass
class RealtimeAudioInputEvent(PluginBaseEvent):
    """Event emitted when audio input is sent to realtime session."""

    type: str = field(default="plugin.realtime_audio_input", init=False)
    data: PcmData | None = None


@dataclass
class RealtimeAudioOutputEvent(PluginBaseEvent):
    """Event emitted when audio output is received from realtime session."""

    type: str = field(default="plugin.realtime_audio_output", init=False)
    data: PcmData | None = None
    response_id: str | None = None


@dataclass
class RealtimeResponseEvent(PluginBaseEvent):
    """Event emitted when realtime session provides a response."""

    type: str = field(default="plugin.realtime_response", init=False)
    original: str | None = None
    text: str | None = None
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_complete: bool = True
    conversation_item_id: str | None = None


@dataclass
class RealtimeConversationItemEvent(PluginBaseEvent):
    """Event emitted for conversation item updates in realtime session."""

    type: str = field(default="plugin.realtime_conversation_item", init=False)
    item_id: str | None = None
    item_type: str | None = None  # "message", "function_call", "function_call_output"
    status: str | None = None  # "completed", "in_progress", "incomplete"
    role: str | None = None  # "user", "assistant", "system"
    content: list[dict[str, Any]] | None = None


@dataclass
class RealtimeErrorEvent(PluginBaseEvent):
    """Event emitted when a realtime error occurs."""

    type: str = field(default="plugin.realtime_error", init=False)
    error: Exception | None = None
    error_code: str | None = None
    context: str | None = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class LLMResponseChunkEvent(PluginBaseEvent):
    type: str = field(default="plugin.llm_response_chunk", init=False)
    content_index: int | None = None
    """The index of the content part that the text delta was added to."""

    delta: str | None = None
    """The text delta that was added."""

    item_id: str | None = None
    """The ID of the output item that the text delta was added to."""

    output_index: int | None = None
    """The index of the output item that the text delta was added to."""

    sequence_number: int | None = None
    """The sequence number for this event."""


@dataclass
class LLMResponseCompletedEvent(PluginBaseEvent):
    """Event emitted after an LLM response is processed."""

    type: str = field(default="plugin.llm_response_completed", init=False)
    original: Any = None
    text: str = ""
    item_id: str | None = None


@dataclass
class ToolStartEvent(PluginBaseEvent):
    """Event emitted when a tool execution starts."""

    type: str = field(default="plugin.llm.tool.start", init=False)
    tool_name: str = ""
    arguments: dict[str, Any] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolEndEvent(PluginBaseEvent):
    """Event emitted when a tool execution ends."""

    type: str = field(default="plugin.llm.tool.end", init=False)
    tool_name: str = ""
    success: bool = True
    result: Any | None = None
    error: str | None = None
    tool_call_id: str | None = None
    execution_time_ms: float | None = None


@dataclass
class RealtimeUserSpeechTranscriptionEvent(PluginBaseEvent):
    """Event emitted when user speech transcription is available from realtime session."""

    type: str = field(default="plugin.realtime_user_speech_transcription", init=False)
    text: str = ""
    original: Any | None = None


@dataclass
class RealtimeAgentSpeechTranscriptionEvent(PluginBaseEvent):
    """Event emitted when agent speech transcription is available from realtime session."""

    type: str = field(default="plugin.realtime_agent_speech_transcription", init=False)
    text: str = ""
    original: Any | None = None
