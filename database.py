"""
Database module for managing image/video metadata and embeddings.
Uses SQLite for persistent storage with optimized batch operations.
"""

import hashlib
import sqlite3
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import logging
import pandas as pd
from build_faiss_index import build_faiss_from_db
from image_processing.cluster_trip_processor import find_sessions, build_similarity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "databases/media_database.db"
FAISS_INDEX_PATH = "databases/faiss_index.bin"
FAISS_META_PATH = "databases/faiss_meta.pkl"

# Embedding module will be lazy-loaded inside process_folder to avoid
# heavy imports (clip/torch/cv2) at module import time which can break
# notebook imports when those packages are not available.


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Initialize database with required tables.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLite connection object
    """
    # Ensure database directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Table 1: All Media
    conn.execute("""
        CREATE TABLE IF NOT EXISTS all_media (
            file_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            type TEXT NOT NULL CHECK(type IN ('image', 'video')),
            date_taken TEXT,
            filesize INTEGER
        )
    """)
    
    # Table 2: All Media Embeddings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS all_media_embeddings (
            file_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (file_id) REFERENCES all_media(file_id) ON DELETE CASCADE
        )
    """)

    # Table 3: All Media Clusters
    conn.execute("""
        CREATE TABLE IF NOT EXISTS all_media_clusters (
            cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            cluster_label INTEGER NOT NULL,
            session_id INTEGER,
            cluster_method TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES all_media(file_id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_all_media_clusters_file_id ON all_media_clusters(file_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_all_media_clusters_label ON all_media_clusters(cluster_label)")
    
    # Index for faster lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON all_media(file_path)")
    
    conn.commit()
    logger.info(f"Database initialized at {db_path}")
    return conn


def normalize_path(file_path: str) -> str:
    """
    Normalize a file path for consistent storage and lookup.
    """
    return os.path.normpath(os.path.abspath(file_path))


def get_existing_paths(conn: sqlite3.Connection) -> set:
    """
    Get all existing file paths from database in one query.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Set of existing file paths
    """
    cursor = conn.execute("SELECT file_path FROM all_media")
    return {normalize_path(row[0]) for row in cursor.fetchall()}


def get_media_type(file_path: str) -> str:
    """
    Determine if file is image or video based on extension.
    
    Args:
        file_path: Path to media file
        
    Returns:
        'image' or 'video'
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    ext = Path(file_path).suffix.lower()
    if ext in image_exts:
        return 'image'
    elif ext in video_exts:
        return 'video'
    return 'image'  # default


def scan_folder(folder_path: str, extensions: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
    """
    Scan folder for media files.
    
    Args:
        folder_path: Path to folder to scan
        extensions: List of extensions to include (e.g., ['.jpg', '.png'])
        
    Returns:
        List of tuples: (file_path, file_name, filesize)
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif',
                      '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    extensions = {ext.lower() for ext in extensions}
    media_files = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in extensions:
                file_path = normalize_path(os.path.join(root, file))
                try:
                    filesize = os.path.getsize(file_path)
                    media_files.append((file_path, file, filesize))
                except OSError as e:
                    logger.warning("Cannot access file %s: %s", file_path, e)
    
    logger.info("Found %d media files in %s", len(media_files), folder_path)
    return media_files


def generate_file_id(file_path: str) -> str:
    """
    Generate a stable unique file_id based on the normalized path and modification time.
    
    Args:
        file_path: Path to media file
        
    Returns:
        Unique file_id string
    """
    normalized_path = normalize_path(file_path)
    mtime = int(os.path.getmtime(normalized_path))
    hash_input = f"{normalized_path}|{mtime}".encode('utf-8')
    return hashlib.sha1(hash_input).hexdigest()


def load_image_embeddings_for_clustering(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load all image records with embeddings for clustering.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame containing file_id, date_taken, and embedding arrays
    """
    cursor = conn.execute(
        """
            SELECT m.file_id, m.date_taken, e.embedding
            FROM all_media m
            JOIN all_media_embeddings e ON m.file_id = e.file_id
            WHERE m.type = 'image'
        """
    )
    rows = cursor.fetchall()

    records = []
    for file_id, date_taken, emb_bytes in rows:
        if emb_bytes is not None:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            records.append({
                'file_id': file_id,
                'date_taken': date_taken,
                'embedings': emb
            })

    if not records:
        return pd.DataFrame(columns=['file_id', 'date_taken', 'embedings'])

    return pd.DataFrame(records)


def update_metadata(conn: sqlite3.Connection, new_files: List[Tuple[str, str, int]]) -> Dict[str, str]:
    """
    Update metadata for new media files.
    
    Args:
        conn: SQLite connection
        new_files: List of tuples (file_path, file_name, filesize)
        
    Returns:
        Dictionary mapping normalized file_path to file_id
    """
    if not new_files:
        logger.info("No new files to update metadata for")
        return {}
    
    logger.info(f"Updating metadata for {len(new_files)} new files")
    
    media_rows = []
    file_id_map = {}
    for file_path, file_name, filesize in new_files:
        normalized_path = normalize_path(file_path)
        file_id = generate_file_id(normalized_path)
        media_type = get_media_type(normalized_path)
        date_taken = datetime.fromtimestamp(os.path.getmtime(normalized_path), tz=timezone.utc).isoformat()
        media_rows.append((file_id, file_name, normalized_path, media_type, date_taken, filesize))
        file_id_map[normalized_path] = file_id
    
    # Insert metadata directly
    conn.executemany(
        """
            INSERT OR IGNORE INTO all_media (file_id, file_name, file_path, type, date_taken, filesize)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
        media_rows
    )
    conn.commit()
    logger.info(f"Metadata updated for {len(media_rows)} files")
    return file_id_map


def get_files_needing_embeddings(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    """
    Get list of all files that have metadata but no embeddings yet.
    Queries directly from database instead of relying on file_id_map.
    This enables resume capability across multiple runs.
    
    Args:
        conn: SQLite connection
        
    Returns:
        List of tuples (file_path, file_id) for files needing embeddings
    """
    # Get all files with embeddings
    cursor = conn.execute("SELECT file_id FROM all_media_embeddings")
    files_with_embeddings = {row[0] for row in cursor.fetchall()}
    
    # Get all files from metadata
    cursor = conn.execute("SELECT file_path, file_id FROM all_media")
    files_needing_embeddings = []
    for file_path, file_id in cursor.fetchall():
        if file_id not in files_with_embeddings:
            files_needing_embeddings.append((file_path, file_id))
    
    return files_needing_embeddings


def update_embeddings(conn: sqlite3.Connection, new_files: List[Tuple[str, str, int]], file_id_map: Dict[str, str], embedding_module=None, batch_size: int = 50) -> None:
    """
    Update embeddings for new media files (images and videos).
    Supports resume: skips files that already have embeddings in database.
    Saves embeddings immediately per file for fault tolerance.
    
    Args:
        conn: SQLite connection
        new_files: List of tuples (file_path, file_name, filesize)
        file_id_map: Dictionary mapping normalized file_path to file_id
        embedding_module: EmbeddingModule instance
        batch_size: Number of files to process in each batch
    """
    if embedding_module is None:
        logger.info("No embedding module available for embedding update")
        return
    
    # Filter to only files that need embeddings (haven't been processed yet)
    # Query all files from database, not just the newly added ones
    media_files = get_files_needing_embeddings(conn)
    
    if not media_files:
        logger.info("All files already have embeddings. Skipping embedding update.")
        return
    
    logger.info(f"Generating embeddings for {len(media_files)} files without embeddings")
    
    def _extract_video_embedding(result):
        if result is None:
            return None

        emb = None
        if isinstance(result, dict):
            # Avoid using `or` which triggers truth-value checks on tensors
            emb = result.get('video_embedding')
            if emb is None:
                emb = result.get('embedding')
        else:
            emb = result

        # Ensure tensor is on CPU before numpy conversion
        try:
            import torch as _torch
            if isinstance(emb, _torch.Tensor) and emb.device.type != 'cpu':
                emb = emb.cpu()
        except Exception:
            # If torch not available or other issue, fall through and return emb as-is
            pass

        return emb
    
    # Process embeddings in batches
    for i in range(0, len(media_files), batch_size):
        batch = media_files[i:i+batch_size]
        image_items = []
        video_items = []
        
        for file_path, file_id in batch:
            try:
                if get_media_type(file_path) == 'video':
                    video_items.append((file_path, file_id))
                else:
                    image_items.append((file_path, file_id))
            except Exception as e:
                logger.error(f"Error determining media type for {file_path}: {e}")
                continue
        
        # Process images in this batch
        if image_items:
            image_paths = [file_path for file_path, _ in image_items]
            try:
                image_embeddings = embedding_module.embed_images_batch(image_paths, batch_size=batch_size)
                for (file_path, file_id), emb in zip(image_items, image_embeddings):
                    try:
                        if emb is not None:
                            emb_bytes = emb.float().numpy().tobytes()
                            # Save immediately per file for fault tolerance
                            conn.execute(
                                "INSERT OR REPLACE INTO all_media_embeddings (file_id, embedding) VALUES (?, ?)",
                                (file_id, emb_bytes)
                            )
                            conn.commit()
                        else:
                            logger.warning(f"Failed to generate embedding for image {file_path}")
                    except Exception as e:
                        logger.error(f"Error saving embedding for {file_path}: {e}")
                        conn.rollback()
                        continue
            except Exception as e:
                logger.error(f"Error processing image batch {i//batch_size}: {e}")
        
        # Process videos in this batch
        if video_items:
            for file_path, file_id in video_items:
                try:
                    video_results = embedding_module.embed_videos_batch([file_path])
                    if video_results:
                        result = video_results[0]
                        emb = _extract_video_embedding(result)
                        if emb is not None:
                            emb_bytes = emb.float().numpy().tobytes()
                            # Save immediately per file for fault tolerance
                            conn.execute(
                                "INSERT OR REPLACE INTO all_media_embeddings (file_id, embedding) VALUES (?, ?)",
                                (file_id, emb_bytes)
                            )
                            conn.commit()
                        else:
                            logger.warning(f"Failed to generate embedding for video {file_path}")
                except Exception as e:
                    logger.error(f"Error processing video {file_path}: {e}")
                    conn.rollback()
                    continue
        
        logger.info(f"Processed embeddings {min(i + len(batch), len(media_files))}/{len(media_files)}")
    
    logger.info("Embeddings updated successfully")


def update_clusters(conn: sqlite3.Connection) -> None:
    """
    Update cluster assignments for all images with embeddings.
    
    Args:
        conn: SQLite connection
    """
    logger.info("Refreshing media clusters")
    
    df = load_image_embeddings_for_clustering(conn)
    if df.empty:
        logger.info("No image embeddings available for clustering")
        return

    df = find_sessions(df)
    labels = build_similarity(df)
    df['cluster_label'] = labels

    cluster_rows = [
        (row.file_id, int(row.cluster_label), int(row.session_id), 'trip')
        for row in df.itertuples(index=False)
    ]

    conn.execute("DELETE FROM all_media_clusters")
    if cluster_rows:
        conn.executemany(
            """
                INSERT INTO all_media_clusters (file_id, cluster_label, session_id, cluster_method)
                VALUES (?, ?, ?, ?)
            """,
            cluster_rows
        )
    conn.commit()
    logger.info("Clusters updated successfully")


def process_folder(
    folder_path: str,
    db_path: str = DB_PATH,
    batch_size: int = 50
) -> Tuple[int, int]:
    """
    Process folder and add new media to database with embeddings and clusters.
    
    Args:
        folder_path: Path to folder containing media files
        embedding_module: Instance of EmbeddingModule from embeding_processor
        db_path: Path to SQLite database
        batch_size: Number of files to process in each batch
        
    Returns:
        Tuple of (new_files_added, total_files_in_db)
    """
    logger.info(f"Starting folder processing for: {folder_path}")
    
    conn = init_db(db_path)

    # Try to lazy-load embedding module only when processing is needed.
    emb_module = None
    try:
        from image_processing.embeding_processor import EmbeddingModule
        emb_module = EmbeddingModule()
        emb_module.load_model()
    except Exception as e:
        logger.warning("Embedding module not available or failed to load: %s", e)
    
    # Get existing paths to skip
    existing_paths = get_existing_paths(conn)
    logger.info(f"Database contains {len(existing_paths)} existing files")
    
    # Scan folder for media files
    media_files = scan_folder(folder_path)
    
    # Filter to only new files
    new_files = [(path, name, size) for path, name, size in media_files if path not in existing_paths]
    logger.info(f"Found {len(new_files)} new files to process")
    
    # Step 1: Update metadata for new files
    file_id_map = {}
    if new_files:
        file_id_map = update_metadata(conn, new_files)
    else:
        logger.info("No new files to add to database")
    
    # Step 2: Update embeddings (for ALL files needing embeddings, not just new ones)
    # This enables resume/continuation capability
    update_embeddings(conn, new_files, file_id_map, emb_module, batch_size)
    
    # Step 3: Update clusters (only if we processed embeddings)
    update_clusters(conn)
    
    # Update FAISS index with all embeddings
    logger.info("Building/Updating FAISS index...")
    build_faiss_from_db(conn)

    # Unload model to free GPU/CPU memory if it was loaded
    try:
        if emb_module is not None:
            emb_module.unload_model()
    except Exception:
        pass
    
    cursor = conn.execute("SELECT COUNT(*) FROM all_media")
    total = cursor.fetchone()[0]
    
    logger.info(f"Folder processing completed. Added {len(new_files)} new files. Total in DB: {total}")
    return len(new_files), total


def clear_media_clusters(conn: sqlite3.Connection) -> None:
    """
    Remove all records from the all_media_clusters table.
    """
    conn.execute("DELETE FROM all_media_clusters")
    conn.commit()


def load_table_as_df(table_name: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load a table from the database as a pandas DataFrame.
    
    Args:
        table_name: Name of the table to load
        db_path: Path to the SQLite database file
        
    Returns:
        pandas DataFrame containing the table data
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def load_embeddings_as_dict(table_name: str = 'all_media_embeddings', db_path: str = DB_PATH) -> Dict[str, np.ndarray]:
    """
    Load embeddings from the database table as a dictionary.

    Args:
        table_name: Name of the embeddings table (default: all_media_embeddings)
        db_path: Path to the SQLite database file

    Returns:
        Dictionary mapping file_id to embedding numpy array
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(f"SELECT file_id, embedding FROM {table_name}")
    rows = cursor.fetchall()
    conn.close()

    embeddings = {}
    for file_id, emb_bytes in rows:
        if emb_bytes is not None:
            embeddings[file_id] = np.frombuffer(emb_bytes, dtype=np.float32)
    return embeddings


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        added, total = process_folder(folder)
        print(f"Added {added} new files. Total in DB: {total}")
    else:
        print("Usage: python database.py <folder_path>")
        print("\nAdditional functions:")
        print("  - build_faiss_from_db(conn): Build FAISS index from database")
        print("  - load_faiss_index(): Load existing FAISS index")
        print("  - save_faiss_index(index, file_ids): Save FAISS index to disk")