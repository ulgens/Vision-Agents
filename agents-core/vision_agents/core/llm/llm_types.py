from enum import Enum
from typing import Any, Literal, TypedDict, Union

from typing_extensions import NotRequired


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageBytesPart(TypedDict):
    type: Literal["image"]
    data: bytes
    mime_type: NotRequired[str]


class ImageURLPart(TypedDict):
    type: Literal["image"]
    url: str
    mime_type: NotRequired[str]


class AudioPart(TypedDict):
    type: Literal["audio"]
    data: bytes
    mime_type: str
    sample_rate: NotRequired[int]
    channels: NotRequired[int]


class JsonPart(TypedDict):
    type: Literal["json"]
    data: dict[str, Any]


ContentPart = Union[TextPart, ImageBytesPart, ImageURLPart, AudioPart, JsonPart]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(TypedDict):
    role: Role
    content: list[ContentPart]


# =============================
# Normalized response contracts
# =============================


class NormalizedStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class ToolSchema(TypedDict, total=False):
    name: str
    description: NotRequired[str]
    parameters_schema: dict[str, Any]


class ResponseFormat(TypedDict, total=False):
    # JSON Schema to enforce structured output, if supported by provider
    json_schema: dict[str, Any]
    # If true, providers should enforce strict adherence where possible
    strict: NotRequired[bool]


class NormalizedUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    raw_usage: NotRequired[dict[str, Any]]


class NormalizedTextItem(TypedDict, total=False):
    type: Literal["text"]
    text: str
    index: NotRequired[int]


class NormalizedAudioItem(TypedDict, total=False):
    type: Literal["audio"]
    data: bytes
    mime_type: NotRequired[str]
    sample_rate: NotRequired[int]
    channels: NotRequired[int]
    index: NotRequired[int]


class NormalizedImageItem(TypedDict, total=False):
    type: Literal["image"]
    data: NotRequired[bytes]
    url: NotRequired[str]
    mime_type: NotRequired[str]
    index: NotRequired[int]


class NormalizedToolCallItem(TypedDict, total=False):
    type: Literal["tool_call"]
    name: str
    arguments_json: dict[str, Any]
    id: NotRequired[str]  # Provider-specific tool call ID (e.g., for OpenAI, Anthropic)
    thought_signature: NotRequired[
        str
    ]  # Gemini-specific thought signature for multi-turn function calls


class NormalizedToolResultItem(TypedDict, total=False):
    type: Literal["tool_result"]
    name: str
    result_json: dict[str, Any]
    is_error: NotRequired[bool]


NormalizedOutputItem = Union[
    NormalizedTextItem,
    NormalizedAudioItem,
    NormalizedImageItem,
    NormalizedToolCallItem,
    NormalizedToolResultItem,
]


class NormalizedResponse(TypedDict, total=False):
    id: str
    model: str
    status: NormalizedStatus
    output: list[NormalizedOutputItem]
    output_text: NotRequired[str]
    usage: NotRequired[NormalizedUsage]
    metadata: NotRequired[dict[str, Any]]
    warnings: NotRequired[list[str]]
    incomplete_details: NotRequired[dict[str, Any]]
    raw: NotRequired[Any]


__all__ = [
    "TextPart",
    "ImageBytesPart",
    "ImageURLPart",
    "AudioPart",
    "JsonPart",
    "ContentPart",
    "Role",
    "Message",
    "NormalizedStatus",
    "ToolSchema",
    "ResponseFormat",
    "NormalizedUsage",
    "NormalizedTextItem",
    "NormalizedAudioItem",
    "NormalizedImageItem",
    "NormalizedToolCallItem",
    "NormalizedToolResultItem",
    "NormalizedOutputItem",
    "NormalizedResponse",
]
