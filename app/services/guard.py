import re

from app.core.exceptions import PromptInjectionError
from app.schemas import Message

_INJECTION_PATTERNS = re.compile(
    r"ignore (all |previous |prior |above |your )?(instructions|rules|prompt|context|system)"
    r"|forget (everything|all|your instructions|who you are)"
    r"|you are now\b"
    r"|act as (if you are|though you are|a )?"
    r"|pretend (you are|to be|that you)"
    r"|disregard (all |your |previous )?(instructions|rules|training)"
    r"|new (persona|role|instructions|task|identity)"
    r"|<\s*system\s*>"
    r"|\bsystem\s*:",
    flags=re.IGNORECASE,
)

_MAX_MESSAGE_LENGTH = 8_000


def check_messages(messages: list[Message]) -> None:
    for msg in messages:
        if msg.role != "user":
            continue
        if len(msg.content) > _MAX_MESSAGE_LENGTH:
            raise PromptInjectionError(
                f"Message exceeds maximum allowed length of {_MAX_MESSAGE_LENGTH} characters."
            )
        if _INJECTION_PATTERNS.search(msg.content):
            raise PromptInjectionError("Message contains patterns that look like prompt injection.")
