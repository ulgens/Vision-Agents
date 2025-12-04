from dataclasses import dataclass, field

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events import PluginBaseEvent
from typing import Any


@dataclass
class AudioReceivedEvent(PluginBaseEvent):
    """Event emitted when audio is received from a participant."""

    type: str = field(default="plugin.edge.audio_received", init=False)
    pcm_data: PcmData | None = None
    participant: Any | None = None


@dataclass
class TrackAddedEvent(PluginBaseEvent):
    """Event emitted when a track is added to the call."""

    type: str = field(default="plugin.edge.track_added", init=False)
    track_id: str | None = None
    track_type: int | None = None
    user: Any | None = None


@dataclass
class TrackRemovedEvent(PluginBaseEvent):
    """Event emitted when a track is removed from the call."""

    type: str = field(default="plugin.edge.track_removed", init=False)
    track_id: str | None = None
    track_type: int | None = None
    user: Any | None = None


@dataclass
class CallEndedEvent(PluginBaseEvent):
    """Event emitted when a call ends."""

    type: str = field(default="plugin.edge.call_ended", init=False)
    args: tuple | None = None
    kwargs: dict | None = None
