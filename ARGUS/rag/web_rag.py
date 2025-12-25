from __future__ import annotations
import sqlite3
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np
from config_metrics.main_config import get_semantic_model


@dataclass
class WebHit:
    url: str
    title: str
    snippet: str
    score: float
    timestamp: str
    quality: float


class WebRAG:
    """
    WebRAG over the Living-LLM scraper DB (web_learning.db).

    It expects the scraper table:
      scraped_content(url, title, content, timestamp, quality_score, content_hash, ...)
    (matches your intelligent_scraper.py schema)

    It adds:
      rag_chunks(chunk_hash, url, title, chunk, timestamp, quality_score, content_hash)
      rag_embeddings(chunk_hash, dim, embedding BLOB)
    """

    def __init__(
        self,
        scraper_db_path: str,
        min_quality: float = 0.55,
        max_chunk_chars: int = 1100,
        overlap_chars: int = 150,
        lookback_days: int = 30,
        ingest_batch: int = 200,
    ):
        self.db_path = scraper_db_path
        self.min_quality = float(min_quality)
        self.max_chunk_chars = int(max_chunk_chars)
        self.overlap_chars = int(overlap_chars)
        self.lookback_days = int(lookback_days)
        self.ingest_batch = int(ingest_batch)

        self.model = get_semantic_model()
        self._init_tables()

    # ---------- DB ----------
    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30)
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA busy_timeout=8000;")
        return c

    def _init_tables(self) -> None:
        conn = self._conn()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_hash TEXT PRIMARY KEY,
                content_hash TEXT,
                url TEXT,
                title TEXT,
                chunk TEXT,
                timestamp TEXT,
                quality_score REAL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_time ON rag_chunks(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_chunks_quality ON rag_chunks(quality_score)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_embeddings (
                chunk_hash TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )

        conn.commit()
        conn.close()

    # ---------- Helpers ----------
    def _hash(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

    def _chunk(self, text: str) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []
        out: List[str] = []
        step = max(1, self.max_chunk_chars - self.overlap_chars)
        i = 0
        while i < len(t):
            out.append(t[i : i + self.max_chunk_chars])
            i += step
        return out

    def _to_blob(self, vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    def _from_blob(self, blob: bytes, dim: int) -> np.ndarray:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size != dim:
            # corrupted or mismatched
            return v[:dim] if v.size > dim else np.pad(v, (0, dim - v.size))
        return v

    def _embed(self, texts: List[str]) -> np.ndarray:
        # sentence-transformers style
        emb = self.model.encode(texts, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        # normalize for cosine via dot
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb / norms

    # ---------- Ingest ----------
    def ingest_new(self) -> int:
        """
        Pull recent scraped_content rows, chunk them, and insert into rag_chunks.
        Only inserts new chunk_hashes.
        """
        cutoff = (datetime.utcnow() - timedelta(days=self.lookback_days)).isoformat()

        conn = self._conn()
        cur = conn.cursor()

        # read recent high-quality articles
        cur.execute(
            """
            SELECT url, title, content, timestamp, quality_score, content_hash
            FROM scraped_content
            WHERE quality_score >= ?
              AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (self.min_quality, cutoff, self.ingest_batch),
        )
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return 0

        inserted = 0
        for url, title, content, ts, qs, content_hash in rows:
            for ch in self._chunk(content or ""):
                chunk_hash = self._hash((content_hash or "") + "||" + ch + "||" + (url or ""))
                cur.execute(
                    """
                    INSERT OR IGNORE INTO rag_chunks
                    (chunk_hash, content_hash, url, title, chunk, timestamp, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_hash, content_hash, url, title, ch, ts, float(qs or 0.0)),
                )
                if cur.rowcount > 0:
                    inserted += 1

        conn.commit()
        conn.close()
        return inserted

    # ---------- Retrieval ----------
    def retrieve(self, query: str, top_k: int = 4, scan_limit: int = 600) -> str:
        """
        Returns a formatted WEB_MEMORY string (title + url + snippet).
        """
        q = (query or "").strip()
        if not q:
            return ""

        # keep index fresh
        self.ingest_new()

        conn = self._conn()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT chunk_hash, url, title, chunk, timestamp, quality_score
            FROM rag_chunks
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (int(scan_limit),),
        )
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return ""

        qv = self._embed([q])[0]  # normalized

        # load cached embeddings
        chunk_hashes = [r[0] for r in rows]
        placeholders = ",".join("?" for _ in chunk_hashes)

        cached = {}
        if chunk_hashes:
            cur.execute(
                f"""
                SELECT chunk_hash, dim, embedding
                FROM rag_embeddings
                WHERE chunk_hash IN ({placeholders})
                """,
                chunk_hashes,
            )
            for h, dim, blob in cur.fetchall():
                cached[h] = self._from_blob(blob, int(dim))

        # compute missing embeddings in batches
        missing: List[Tuple[str, str]] = []
        for (h, url, title, chunk, ts, qs) in rows:
            if h not in cached:
                missing.append((h, chunk))

        if missing:
            # embed in chunks
            texts = [t for _, t in missing]
            embs = self._embed(texts)
            now = time.time()
            for (h, _), ev in zip(missing, embs):
                cached[h] = ev
                cur.execute(
                    """
                    INSERT OR REPLACE INTO rag_embeddings (chunk_hash, dim, embedding, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (h, int(ev.shape[0]), self._to_blob(ev), now),
                )
            conn.commit()

        # score
        hits: List[WebHit] = []
        for (h, url, title, chunk, ts, qs) in rows:
            ev = cached.get(h)
            if ev is None or ev.size == 0:
                continue
            score = float(np.dot(qv, ev))  # cosine (normalized)
            snippet = " ".join((chunk or "").split())
            if len(snippet) > 380:
                snippet = snippet[:380].rsplit(" ", 1)[0] + "..."
            hits.append(
                WebHit(
                    url=url or "",
                    title=(title or "").strip(),
                    snippet=snippet,
                    score=score,
                    timestamp=ts or "",
                    quality=float(qs or 0.0),
                )
            )

        conn.close()

        if not hits:
            return ""

        hits.sort(key=lambda x: x.score, reverse=True)
        hits = hits[: int(top_k)]

        # format WEB_MEMORY block
        lines: List[str] = []
        for i, h in enumerate(hits, 1):
            title = h.title if h.title else "Untitled"
            lines.append(
                f"{i}) {title}\n"
                f"   URL: {h.url}\n"
                f"   Time: {h.timestamp} | Quality: {h.quality:.2f} | Score: {h.score:.3f}\n"
                f"   Snippet: {h.snippet}"
            )

        return "\n".join(lines)