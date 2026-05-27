import asyncio
from collections import defaultdict

_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)


async def subscribe(camera_id: str):
    q: asyncio.Queue = asyncio.Queue()
    _subscribers[camera_id].append(q)
    try:
        while True:
            yield await q.get()
    finally:
        _subscribers[camera_id].remove(q)
        if not _subscribers[camera_id]:
            del _subscribers[camera_id]


async def broadcast(camera_id: str, data: dict) -> None:
    for q in list(_subscribers.get(camera_id, [])):
        await q.put(data)
