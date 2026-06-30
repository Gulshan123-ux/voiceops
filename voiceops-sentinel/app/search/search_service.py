from __future__ import annotations
import json
import sqlite3
from typing import Optional
from app.database import get_db_connection

def search_calls(
    query: Optional[str] = None,
    sentiment: Optional[str] = None,
    flagged: Optional[bool] = None,
    tag: Optional[str] = None,
    wer_min: Optional[float] = None,
    wer_max: Optional[float] = None,
    duration_min: Optional[float] = None,
    duration_max: Optional[float] = None,
) -> list[dict]:
    """
    Search and filter processed call records stored in the SQLite database.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        sql = "SELECT * FROM calls"
        conditions = []
        params = []

        if query:
            # Match query against filename, transcript, redacted_transcript, or action_items
            conditions.append(
                "(filename LIKE ? OR transcript LIKE ? OR redacted_transcript LIKE ? OR action_items LIKE ?)"
            )
            like_query = f"%{query}%"
            params.extend([like_query, like_query, like_query, like_query])

        if sentiment:
            conditions.append("sentiment = ?")
            params.append(sentiment)

        if flagged is not None:
            conditions.append("flagged = ?")
            params.append(1 if flagged else 0)

        if tag:
            # Tags are stored as a JSON array, e.g. ["resolved", "billing"]
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        if wer_min is not None:
            conditions.append("wer_score >= ?")
            params.append(wer_min)

        if wer_max is not None:
            conditions.append("wer_score <= ?")
            params.append(wer_max)

        if duration_min is not None:
            conditions.append("duration >= ?")
            params.append(duration_min)

        if duration_max is not None:
            conditions.append("duration <= ?")
            params.append(duration_max)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY created_at DESC"

        cursor.execute(sql, params)
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

            tags_data = []
            if row_dict.get("tags"):
                try:
                    tags_data = json.loads(row_dict["tags"])
                except Exception:
                    pass

            latency_data = None
            if row_dict.get("latency_report"):
                try:
                    latency_data = json.loads(row_dict["latency_report"])
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
                "summary": row_dict.get("summary") or "",
                "summary_issue": row_dict.get("summary_issue") or "",
                "summary_resolution": row_dict.get("summary_resolution") or "",
                "summary_follow_up": row_dict.get("summary_follow_up") or "None",
                "summary_engine": row_dict.get("summary_engine") or "extractive",
                "latency_report": latency_data,
                "tags": tags_data,
            }
            calls.append(call_dict)
        return calls
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Search calls failed: {e}")
        return []
    finally:
        conn.close()
