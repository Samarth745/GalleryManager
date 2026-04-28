"""
Database module for managing image/video metadata and embeddings.
Uses SQLite for persistent storage with optimized batch operations.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "media_database.db"


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Initialize database with required tables.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLite connection object
    """
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
    
    # Index for faster lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON all_media(file_path)")
    
    conn.commit()
    logger.info(f"Database initialized at {db_path}")
    return conn


def get_existing_paths(conn: sqlite3.Connection) -> set:
    """
    Get all existing file paths from database in one query.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Set of existing file paths
    """
    cursor = conn.execute("SELECT file_path FROM all_media")
    return {row[0] for row in cursor.fetchall()}


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
                file_path = os.path.join(root, file)
                try:
                    filesize = os.path.getsize(file_path)
                    media_files.append((file_path, file, filesize))
                except OSError as e:
                    logger.warning(f"Cannot access file {file_path}: {e}")
    
    logger.info(f"Found {len(media_files)} media files in {folder_path}")
    return media_files


def generate_file_id(file_path: str) -> str:
    """
    Generate unique file_id based on path and modification time.
    
    Args:
        file_path: Path to media file
        
    Returns:
        Unique file_id string
    """
    mtime = int(os.path.getmtime(file_path))
    return f"{Path(file_path).stem}_{mtime}"


def process_folder(
    folder_path: str,
    embedding_module,
    db_path: str = DB_PATH,
    batch_size: int = 50
) -> Tuple[int, int]:
    """
    Process folder and add new media to database with embeddings.
    
    Args:
        folder_path: Path to folder containing media files
        embedding_module: Instance of EmbeddingModule from embeding_processor
        db_path: Path to SQLite database
        batch_size: Number of files to process in each batch
        
    Returns:
        Tuple of (new_files_added, total_files_in_db)
    """
    conn = init_db(db_path)
    
    # Get existing paths to skip
    existing_paths = get_existing_paths(conn)
    logger.info(f"Database contains {len(existing_paths)} existing files")
    
    # Scan folder for media files
    media_files = scan_folder(folder_path)
    
    # Filter to only new files
    new_files = [(path, name, size) for path, name, size in media_files if path not in existing_paths]
    logger.info(f"Found {len(new_files)} new files to add")
    
    if not new_files:
        cursor = conn.execute("SELECT COUNT(*) FROM all_media")
        total = cursor.fetchone()[0]
        return 0, total
    
    # Prepare batch data for all_media table
    media_rows = []
    for file_path, file_name, filesize in new_files:
        file_id = generate_file_id(file_path)
        media_type = get_media_type(file_path)
        date_taken = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        media_rows.append((file_id, file_name, file_path, media_type, date_taken, filesize))
    
    # Insert media records in batch
    conn.executemany("""
        INSERT OR IGNORE INTO all_media (file_id, file_name, file_path, type, date_taken, filesize)
        VALUES (?, ?, ?, ?, ?, ?)
    """, media_rows)
    conn.commit()
    
    # Get image files for embedding (videos skipped for now)
    image_files = [row[2] for row in media_rows if row[3] == 'image']
    
    if image_files and embedding_module is not None:
        logger.info(f"Generating embeddings for {len(image_files)} images...")
        
        # Process embeddings in batches
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            embeddings = embedding_module.embed_images_batch(batch)
            
            # Prepare embedding rows
            emb_rows = []
            for file_path, emb in zip(batch, embeddings):
                if emb is not None:
                    file_id = generate_file_id(file_path)
                    emb_bytes = emb.numpy().tobytes()
                    emb_rows.append((file_id, emb_bytes))
            
            # Insert embeddings in batch
            if emb_rows:
                conn.executemany("""
                    INSERT OR REPLACE INTO all_media_embeddings (file_id, embedding)
                    VALUES (?, ?)
                """, emb_rows)
                conn.commit()
            
            logger.info(f"Processed embeddings {i+len(batch)}/{len(image_files)}")
    
    cursor = conn.execute("SELECT COUNT(*) FROM all_media")
    total = cursor.fetchone()[0]
    
    return len(new_files), total


# Convenience function for simple usage
def add_folder(folder_path: str, embedding_module=None) -> Tuple[int, int]:
    """
    Add all media from folder to database.
    
    Args:
        folder_path: Path to folder containing media
        embedding_module: EmbeddingModule instance (optional)
        
    Returns:
        Tuple of (new_files_added, total_files_in_db)
    """
    return process_folder(folder_path, embedding_module)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        from image_processing.embeding_processor import EmbeddingModule
        
        folder = sys.argv[1]
        emb_module = EmbeddingModule()
        emb_module.load_model()
        
        added, total = add_folder(folder, emb_module)
        print(f"Added {added} new files. Total in DB: {total}")
        
        emb_module.unload_model()
    else:
        print("Usage: python database.py <folder_path>")