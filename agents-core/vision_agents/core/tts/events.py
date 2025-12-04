import uuid

from getstream.video.rtc import PcmData

from vision_agents.core.events import PluginBaseEvent, ConnectionState
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TTSAudioEvent(PluginBaseEvent):
    """Event emitted when TTS audio data is available."""

    type: str = field(default="plugin.tts_audio", init=False)
    data: PcmData | None = None
    chunk_index: int = 0
    is_final_chunk: bool = True
    text_source: str | None = None
    synthesis_id: str | None = None


@dataclass
class TTSSynthesisStartEvent(PluginBaseEvent):
    """Event emitted when TTS synthesis begins."""

    type: str = field(default="plugin.tts_synthesis_start", init=False)
    text: str | None = None
    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str | None = None
    voice_id: str | None = None
    estimated_duration_ms: float | None = None


@dataclass
class TTSSynthesisCompleteEvent(PluginBaseEvent):
    """Event emitted when TTS synthesis completes."""

    type: str = field(default="plugin.tts_synthesis_complete", init=False)
    synthesis_id: str | None = None
    text: str | None = None
    total_audio_bytes: int = 0
    synthesis_time_ms: float = 0.0
    audio_duration_ms: float | None = None
    chunk_count: int = 1
    real_time_factor: float | None = None


@dataclass
class TTSErrorEvent(PluginBaseEvent):
    """Event emitted when a TTS error occurs."""

    type: str = field(default="plugin.tts_synthesis_error", init=False)
    error: Exception | None = None
    error_code: str | None = None
    context: str | None = None
    text_source: str | None = None
    synthesis_id: str | None = None
    is_recoverable: bool = True

    @property
    def error_message(self) -> str:
        return str(self.error) if self.error else "Unknown error"


@dataclass
class TTSConnectionEvent(PluginBaseEvent):
    """Event emitted for TTS connection state changes."""

    type: str = field(default="plugin.tts_connection", init=False)
    connection_state: ConnectionState | None = None
    provider: str | None = None
    details: dict[str, Any] | None = None
