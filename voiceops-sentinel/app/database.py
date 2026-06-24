"""
app/database.py
SQLite Database helper module for VoiceOps Sentinel.
No ORM, pure sqlite3.
"""

from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "./voiceops.db")


def get_db_connection() -> sqlite3.Connection:
    """Get connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialise the database tables if they do not exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                duration REAL NOT NULL,
                transcript TEXT NOT NULL,
                redacted_transcript TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                wer_score REAL,
                action_items TEXT NOT NULL,
                flagged INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        # Check if segments column exists, if not add it dynamically
        cursor.execute("PRAGMA table_info(calls)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "segments" not in columns:
            cursor.execute("ALTER TABLE calls ADD COLUMN segments TEXT")
            logger.info("Added 'segments' column to calls table")

        conn.commit()
        logger.info(f"Database initialized successfully at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
    finally:
        conn.close()


def insert_call(
    job_id: str,
    filename: str,
    duration: float,
    transcript: str,
    redacted_transcript: str,
    sentiment: str,
    sentiment_score: float,
    wer_score: float | None,
    action_items: list[str],
    flagged: bool,
    segments: list | None = None,
) -> None:
    """Insert a new call analytics record."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Serialize segments if present
        segments_json = None
        if segments:
            serialized_segs = []
            for seg in segments:
                if hasattr(seg, "model_dump"):
                    serialized_segs.append(seg.model_dump())
                elif hasattr(seg, "dict"):
                    serialized_segs.append(seg.dict())
                else:
                    serialized_segs.append(seg)
            segments_json = json.dumps(serialized_segs)

        cursor.execute(
            """
            INSERT OR REPLACE INTO calls (
                id, filename, duration, transcript, redacted_transcript,
                sentiment, sentiment_score, wer_score, action_items, flagged, created_at, segments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                filename,
                duration,
                transcript,
                redacted_transcript,
                sentiment,
                sentiment_score,
                wer_score,
                json.dumps(action_items),
                1 if flagged else 0,
                created_at,
                segments_json,
            ),
        )
        conn.commit()
        logger.info(f"Call record inserted: {job_id}")
    except Exception as e:
        logger.error(f"Failed to insert call record: {e}")
    finally:
        conn.close()


def get_all_calls() -> list[dict]:
    """Retrieve all call records sorted by created_at DESC."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM calls ORDER BY created_at DESC")
        rows = cursor.fetchall()
        calls = []
        for row in rows:
            row_dict = dict(row)

            segments_data = []
            if row_dict.get("segments"):
                try:
                    segments_data = json.loads(row_dict["segments"])
                except Exception:
                    pass

            call_dict = {
                "job_id": row_dict["id"],
                "audio_file": row_dict["filename"],
                "duration_seconds": row_dict["duration"],
                "full_transcript": row_dict["transcript"],
                "redacted_transcript": row_dict["redacted_transcript"],
                "sentiment": row_dict["sentiment"],
                "sentiment_score": row_dict["sentiment_score"],
                "wer_score": row_dict["wer_score"],
                "action_items": json.loads(row_dict["action_items"]),
                "flagged": bool(row_dict["flagged"]),
                "processed_at": row_dict["created_at"],
                "segments": segments_data,
            }
            calls.append(call_dict)
        return calls
    except Exception as e:
        logger.error(f"Failed to query all calls: {e}")
        return []
    finally:
        conn.close()


def get_call_by_id(job_id: str) -> dict | None:
    """Retrieve a specific call record by ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM calls WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            row_dict = dict(row)

            segments_data = []
            if row_dict.get("segments"):
                try:
                    segments_data = json.loads(row_dict["segments"])
                except Exception:
                    pass

            return {
                "job_id": row_dict["id"],
                "audio_file": row_dict["filename"],
                "duration_seconds": row_dict["duration"],
                "full_transcript": row_dict["transcript"],
                "redacted_transcript": row_dict["redacted_transcript"],
                "sentiment": row_dict["sentiment"],
                "sentiment_score": row_dict["sentiment_score"],
                "wer_score": row_dict["wer_score"],
                "action_items": json.loads(row_dict["action_items"]),
                "flagged": bool(row_dict["flagged"]),
                "processed_at": row_dict["created_at"],
                "segments": segments_data,
            }
        return None
    except Exception as e:
        logger.error(f"Failed to query call by id {job_id}: {e}")
        return None
    finally:
        conn.close()


def delete_call_by_id(job_id: str) -> bool:
    """Delete a call record by ID. Returns True if row was deleted."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM calls WHERE id = ?", (job_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to delete call {job_id}: {e}")
        return False
    finally:
        conn.close()


def get_call_stats() -> dict:
    """Compute and return overall dashboard metrics."""
    conn = get_db_connection()
    stats = {
        "total_calls": 0,
        "avg_wer": None,
        "positive_calls": 0,
        "negative_calls": 0,
        "neutral_calls": 0,
        "flagged_calls": 0,
        "avg_duration": 0.0,
    }
    try:
        cursor = conn.cursor()
        
        # Total counts and averages
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total, 
                AVG(duration) as avg_dur,
                SUM(CASE WHEN flagged = 1 THEN 1 ELSE 0 END) as flagged_count,
                SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as pos_count,
                SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as neg_count,
                SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) as neu_count
            FROM calls
            """
        )
        row = cursor.fetchone()
        if row and row["total"] > 0:
            stats["total_calls"] = row["total"]
            stats["avg_duration"] = round(row["avg_dur"], 2) if row["avg_dur"] else 0.0
            stats["flagged_calls"] = row["flagged_count"] or 0
            stats["positive_calls"] = row["pos_count"] or 0
            stats["negative_calls"] = row["neg_count"] or 0
            stats["neutral_calls"] = row["neu_count"] or 0
            
        # Average WER (only of calls that have a WER computed)
        cursor.execute("SELECT AVG(wer_score) as avg_wer FROM calls WHERE wer_score IS NOT NULL")
        wer_row = cursor.fetchone()
        if wer_row and wer_row["avg_wer"] is not None:
            stats["avg_wer"] = round(wer_row["avg_wer"], 4)
            
        return stats
    except Exception as e:
        logger.error(f"Failed to query stats: {e}")
        return stats
    finally:
        conn.close()
