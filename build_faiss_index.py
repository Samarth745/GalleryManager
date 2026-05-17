import os
import sqlite3
import logging
import pickle
from typing import Tuple, List

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "databases/media_database.db"
FAISS_INDEX_PATH = "databases/faiss_index.bin"
FAISS_META_PATH = "databases/faiss_meta.pkl"


def init_faiss_index(embedding_dim: int = 512) -> faiss.Index:
    """
    Initialize a FAISS index for vector storage.
    
    Args:
        embedding_dim: Dimension of the embedding vectors
        
    Returns:
        FAISS Index object
    """
    # Use Inner Product index for cosine similarity (after normalization)
    index = faiss.IndexFlatIP(embedding_dim)
    logger.info(f"FAISS index initialized with dimension {embedding_dim}")
    return index


def save_faiss_index(index: faiss.Index, file_ids: List[str], metadata: dict = None):
    """
    Save FAISS index and metadata to disk.
    
    Args:
        index: FAISS Index object
        file_ids: List of file_ids corresponding to index vectors
        metadata: Optional metadata dictionary
    """
    # Ensure directory exists
    index_dir = os.path.dirname(FAISS_INDEX_PATH)
    if index_dir and not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Save metadata (file_ids mapping)
    meta = {
        'file_ids': file_ids,
        'embedding_dim': index.d,
        'total_vectors': index.ntotal,
        'extra': metadata or {}
    }
    with open(FAISS_META_PATH, 'wb') as f:
        pickle.dump(meta, f)
    
    logger.info(f"FAISS index saved: {index.ntotal} vectors to {FAISS_INDEX_PATH}")


def load_faiss_index() -> Tuple[faiss.Index, dict]:
    """
    Load FAISS index and metadata from disk.
    
    Returns:
        Tuple of (FAISS Index, metadata dict)
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning("FAISS index file not found")
        return None, None
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(FAISS_META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"FAISS index loaded: {index.ntotal} vectors")
    return index, metadata


def build_faiss_from_db(conn: sqlite3.Connection, embedding_dim: int = 512) -> faiss.Index:
    """
    Build FAISS index from existing database embeddings.
    
    Args:
        conn: SQLite connection
        embedding_dim: Dimension of embeddings
        
    Returns:
        FAISS Index with all embeddings
    """
    cursor = conn.execute("SELECT file_id, embedding FROM all_media_embeddings")
    rows = cursor.fetchall()
    
    if not rows:
        logger.warning("No embeddings found in database")
        return init_faiss_index(embedding_dim)
    
    # Build vectors and file_ids
    vectors = []
    file_ids = []
    
    # Detect actual embedding dimension from first row
    actual_dim = None
    for file_id, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        actual_dim = len(emb)
        break
    
    if actual_dim is None:
        logger.warning("No embeddings found in database")
        return init_faiss_index(embedding_dim)
    
    # Rebuild vectors with detected dimension
    vectors = []
    file_ids = []
    for file_id, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        # Normalize for cosine similarity
        emb = emb / np.linalg.norm(emb)
        vectors.append(emb)
        file_ids.append(file_id)
    
    # Create index with detected dimension
    index = faiss.IndexFlatIP(actual_dim)
    vectors_array = np.array(vectors, dtype=np.float32)
    index.add(vectors_array)
    
    # Save to disk
    save_faiss_index(index, file_ids)
    
    logger.info(f"Built FAISS index from {len(file_ids)} embeddings")
    return index

