"""Bound ASGI receive bytes before multipart files can exhaust temporary disk."""
from __future__ import annotations

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app, *, max_bytes=26*1024*1024):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        received, rejected, sent_rejection = 0, False, False

        async def bounded_receive():
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    # This specific exception makes even older Starlette
                    # multipart parsers close all temporary upload files.
                    raise MultiPartException("Request body exceeds upload budget")
            return message

        async def bounded_send(message):
            nonlocal sent_rejection
            if rejected:
                if not sent_rejection:
                    sent_rejection = True
                    response = JSONResponse({"error": {"code": "file_size_limit", "message": "上传请求不得超过26 MiB，单文件不得超过25 MiB。"}}, status_code=413, headers={"Cache-Control": "no-store"})
                    await response(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, bounded_receive, bounded_send)
        except MultiPartException:
            if not rejected:
                raise
        if rejected and not sent_rejection:
            await bounded_send({"type": "http.response.start"})
