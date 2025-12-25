# web_learning/scraper_service.py
from __future__ import annotations

import asyncio
import threading
import time
import random
from dataclasses import replace
from typing import Optional

from .intelligent_scraper import ScrapingConfig, IntelligentScraper


class ScraperService:
    """
    Runs IntelligentScraper on a background thread with its own asyncio loop.
    Writes into the scraper sqlite DB (ScrapingConfig.database_path).

    - start(): launches thread
    - stop(): signals shutdown
    """

    def __init__(
        self,
        db_path: str,
        cycle_seconds: int = 3600,
        jitter_seconds: int = 120,
        enabled: bool = True,
    ):
        self.db_path = db_path
        self.cycle_seconds = int(cycle_seconds)
        self.jitter_seconds = int(jitter_seconds)
        self.enabled = bool(enabled)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def _run(self) -> None:
        asyncio.run(self._async_main())

    async def _async_main(self) -> None:
        # Create config and force DB path
        cfg = ScrapingConfig()
        try:
            # ScrapingConfig is a dataclass, so replace() works
            cfg = replace(cfg, database_path=self.db_path)
        except Exception:
            # if replace fails for any reason, do a direct set
            cfg.database_path = self.db_path

        scraper = IntelligentScraper(cfg)

        try:
            while not self._stop_evt.is_set():
                try:
                    # Scrape feeds (scraper stores into DB internally)
                    await scraper.scrape_news_feeds()
                except Exception:
                    # Don't crash the service; just try again next cycle
                    pass

                # Sleep with jitter so you don't hammer feeds at exactly the same time
                sleep_s = self.cycle_seconds + random.randint(0, max(0, self.jitter_seconds))
                # break into small sleeps so stop() responds fast
                end_t = time.time() + sleep_s
                while time.time() < end_t:
                    if self._stop_evt.is_set():
                        break
                    await asyncio.sleep(0.5)

        finally:
            try:
                await scraper.close()
            except Exception:
                pass