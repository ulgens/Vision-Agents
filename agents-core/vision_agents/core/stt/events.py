from vision_agents.core.events import PluginBaseEvent, ConnectionState
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptResponse:
    confidence: float | None = None
    language: str | None = None
    processing_time_ms: float | None = None
    audio_duration_ms: float | None = None
    model_name: str | None = None
    other: dict | None = None


@dataclass
class STTTranscriptEvent(PluginBaseEvent):
    """Event emitted when a complete transcript is available."""

    type: str = field(default="plugin.stt_transcript", init=False)
    text: str = ""
    response: TranscriptResponse = field(default_factory=TranscriptResponse)

    def __post_init__(self):
        if not self.text:
            raise ValueError("Transcript text cannot be empty")

    # Convenience properties for backward compatibility
    @property
    def confidence(self) -> float | None:
        return self.response.confidence

    @property
    def language(self) -> str | None:
        return self.response.language

    @property
    def processing_time_ms(self) -> float | None:
        return self.response.processing_time_ms

    @property
    def audio_duration_ms(self) -> float | None:
        return self.response.audio_duration_ms

    @property
    def model_name(self) -> str | None:
        return self.response.model_name


@dataclass
class STTPartialTranscriptEvent(PluginBaseEvent):
    """Event emitted when a partial transcript is available."""

    type: str = field(default="plugin.stt_partial_transcript", init=False)
    text: str = ""
    response: TranscriptResponse = field(default_factory=TranscriptResponse)

    # Convenience properties for backward compatibility
    @property
    def confidence(self) -> float | None:
        return self.response.confidence

    @property
    def language(self) -> str | None:
        return self.response.language

    @property
    def processing_time_ms(self) -> float | None:
        return self.response.processing_time_ms

    @property
    def audio_duration_ms(self) -> float | None:
        return self.response.audio_duration_ms

    @property
    def model_name(self) -> str | None:
        return self.response.model_name


@dataclass
class STTErrorEvent(PluginBaseEvent):
    """Event emitted when an STT error occurs."""

    type: str = field(default="plugin.stt_error", init=False)
    error: Exception | None = None
    error_code: str | None = None
    context: str | None = None
    retry_count: int = 0
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class STTConnectionEvent(PluginBaseEvent):
    """Event emitted for STT connection state changes."""

    type: str = field(default="plugin.stt_connection", init=False)
    connection_state: ConnectionState | None = None
    provider: str | None = None
    details: dict[str, Any] | None = None
    reconnect_attempts: int = 0
