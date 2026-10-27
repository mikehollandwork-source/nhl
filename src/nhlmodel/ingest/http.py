"""HTTP helpers with retries (sync + bounded-concurrency async)."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
TIMEOUT = httpx.Timeout(30.0)


def get(url: str, params: dict | None = None, retries: int = 4, headers: dict | None = None) -> httpx.Response:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = httpx.get(url, params=params, timeout=TIMEOUT, follow_redirects=True,
                          headers={"User-Agent": UA, **(headers or {})})
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
            return r
        except (httpx.HTTPError, httpx.HTTPStatusError) as e:
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def get_json(url: str, params: dict | None = None) -> dict | list:
    r = get(url, params)
    r.raise_for_status()
    return r.json()


async def _fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, retries: int = 4):
    async with sem:
        for i in range(retries):
            try:
                r = await client.get(url)
                if r.status_code == 404:
                    return url, None
                r.raise_for_status()
                return url, r.json()
            except (httpx.HTTPError, ValueError) as e:
                if i == retries - 1:
                    log.warning("giving up on %s: %s", url, e)
                    return url, None
                await asyncio.sleep(2 ** i)
    return url, None


async def _fetch_many(urls: list[str], concurrency: int) -> dict[str, dict | None]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": UA}) as c:
        results = await asyncio.gather(*[_fetch_one(c, sem, u) for u in urls])
    return dict(results)


def fetch_many(urls: list[str], concurrency: int = 8) -> dict[str, dict | None]:
    """Fetch many JSON URLs concurrently. Failed URLs map to None."""
    return asyncio.run(_fetch_many(urls, concurrency))
