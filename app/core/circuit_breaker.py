import time
from enum import Enum

from redis.asyncio import Redis

from app.core.exceptions import LLMUnavailableError


class CircuitState(Enum):
    CLOSED = "closed"  # normal — requests pass through
    OPEN = "open"  # too many failures — requests blocked
    HALF_OPEN = "half_open"  # cooldown passed — one request allowed through


class NullCircuitBreaker:
    """No-op circuit breaker for use in tests."""

    async def __aenter__(self) -> "NullCircuitBreaker":
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False


class CircuitBreaker:
    """Redis-backed circuit breaker for external service calls.

    Transitions:
        CLOSED → OPEN      when failure_threshold consecutive failures occur
        OPEN   → HALF_OPEN after recovery_timeout seconds
        HALF_OPEN → CLOSED on success, → OPEN on failure
    """

    def __init__(
        self,
        redis: Redis,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._redis = redis
        self._name = name
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._key_failures = f"cb:{name}:failures"
        self._key_opened_at = f"cb:{name}:opened_at"

    async def state(self) -> CircuitState:
        opened_at_raw = await self._redis.get(self._key_opened_at)
        if opened_at_raw is None:
            return CircuitState.CLOSED
        elapsed = time.time() - float(opened_at_raw)
        if elapsed < self._recovery_timeout:
            return CircuitState.OPEN
        return CircuitState.HALF_OPEN

    async def record_success(self) -> None:
        await self._redis.delete(self._key_failures, self._key_opened_at)

    async def record_failure(self) -> None:
        failures = await self._redis.incr(self._key_failures)
        if int(failures) >= self._threshold:
            await self._redis.set(self._key_opened_at, time.time())

    async def __aenter__(self) -> "CircuitBreaker":
        current_state = await self.state()
        if current_state == CircuitState.OPEN:
            raise LLMUnavailableError(
                f"Circuit '{self._name}' is OPEN — provider temporarily blocked."
            )
        return self

    async def __aexit__(self, exc_type: type | None, exc: BaseException | None, tb: object) -> bool:
        if exc is None:
            await self.record_success()
        elif isinstance(exc, LLMUnavailableError):
            await self.record_failure()
        return False  # never suppress exceptions
