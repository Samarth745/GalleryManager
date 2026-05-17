import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import sqlite3

DB_PATH = Path("databases/media_database.db")


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_media_stats() -> Dict[str, int]:
    conn = _get_db_connection()
    cursor = conn.execute("SELECT COUNT(*) FROM all_media")
    total = cursor.fetchone()[0] if cursor else 0
    image_count = conn.execute("SELECT COUNT(*) FROM all_media WHERE type = 'image'").fetchone()[0]
    video_count = conn.execute("SELECT COUNT(*) FROM all_media WHERE type = 'video'").fetchone()[0]
    embedding_count = conn.execute("SELECT COUNT(*) FROM all_media_embeddings").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "image": image_count,
        "video": video_count,
        "embeddings": embedding_count,
    }


def get_full_media_df() -> pd.DataFrame:
    conn = _get_db_connection()
    df = pd.read_sql_query("SELECT * FROM all_media ORDER BY date_taken DESC", conn)
    conn.close()
    return df


def get_sample_media(sample_size: int = 4) -> List[Dict[str, str]]:
    df = get_full_media_df()
    if df.empty:
        return []

    sample = df.head(sample_size)
    return [
        {
            "file_path": row["file_path"],
            "file_name": row["file_name"],
            "type": row["type"],
        }
        for _, row in sample.iterrows()
    ]


def safe_image_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    return ""
