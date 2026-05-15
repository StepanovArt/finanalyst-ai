import time
import uuid

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        with logger.contextualize(request_id=request_id):
            logger.info("{} {}", request.method, request.url.path)
            response: Response = await call_next(request)  # type: ignore[operator]
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "{} {} {} {:.1f}ms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )

        response.headers["X-Request-ID"] = request_id
        return response
