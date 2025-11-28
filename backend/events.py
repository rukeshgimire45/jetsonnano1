from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator


class EventBus:
    """Simple pub/sub for detection events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def publish(self, payload: dict[str, Any]) -> None:
        await self._queue.put(payload)

    def publish_from_thread(self, payload: dict[str, Any]) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.publish(payload), self._loop)

    async def subscribe(self) -> AsyncGenerator[dict[str, Any], None]:
        while True:
            payload = await self._queue.get()
            yield payload
            self._queue.task_done()


event_bus = EventBus()
