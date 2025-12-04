import asyncio
import datetime
import tempfile
from dataclasses import dataclass, asdict

import aiortc.mediastreams

from ..edge.types import Participant
from ..llm.events import LLMResponseCompletedEvent
from ..utils.video_forwarder import VideoForwarder


@dataclass
class AgentOptions:
    model_dir: str

    def update(self, other: "AgentOptions") -> "AgentOptions":
        merged_dict = asdict(self)

        for key, value in asdict(other).items():
            if value is not None:
                merged_dict[key] = value

        return AgentOptions(**merged_dict)


# Cache tempdir at module load time to avoid blocking I/O during async operations
_DEFAULT_MODEL_DIR = tempfile.gettempdir()


def default_agent_options():
    return AgentOptions(model_dir=_DEFAULT_MODEL_DIR)


@dataclass
class TrackInfo:
    id: str
    type: int
    processor: str
    priority: int  # higher goes first
    participant: Participant | None
    track: aiortc.mediastreams.VideoStreamTrack
    forwarder: VideoForwarder


@dataclass
class LLMTurn:
    input: str
    participant: Participant | None
    started_at: datetime.datetime
    finished_at: datetime.datetime | None = None
    canceled_at: datetime.datetime | None = None
    response: LLMResponseCompletedEvent | None = None
    task: asyncio.Task | None = None
    turn_finished: bool = False
