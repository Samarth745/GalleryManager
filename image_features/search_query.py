import sqlite3
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
# Paths
DB_PATH = "databases/media_database.db"
FAISS_INDEX_PATH = "databases/faiss_index.bin"
FAISS_META_PATH = "databases/faiss_meta.pkl"

# Search threshold (21% = 0.21)
SEARCH_SIMILARITY_THRESHOLD = 0.21

import torch
import clip

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_faiss_index() -> Tuple[faiss.Index, dict]:
    """
    Load FAISS index and metadata.
    
    Returns:
        Tuple of (FAISS Index, metadata dict)
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(FAISS_META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded FAISS index: {index.ntotal} vectors")
    return index, metadata

def embed_text(query: str) -> np.ndarray:
    """
    Embed a text query using CLIP.
    
    Args:
        query: Text query string
    
    Returns:
        Normalized embedding numpy array
    """
    text_tokens = clip.tokenize([query]).to(device)
    
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    
    return text_embedding.cpu().numpy().astype(np.float32)

def load_file_paths_from_db(file_ids: List[str]) -> Dict[str, str]:
    """
    Load file paths from database for given file IDs.
    
    Returns:
        Dictionary mapping file_id -> file_path
    """
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(file_ids)) if file_ids else ""
    
    if not placeholders:
        conn.close()
        return {}
    
    cursor = conn.execute(
        f"SELECT file_id, file_path FROM all_media WHERE file_id IN ({placeholders})",
        file_ids
    )
    path_dict = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return path_dict

def search_by_text(
    query_text: str,
    similarity_threshold: float = 0.1,
    top_k: int = 20
) -> List[dict]:
    """
    Search for similar media using a text query.
    
    Args:
        query_text: Text query string
        index: FAISS Index
        file_ids: List of file IDs corresponding to index vectors
        similarity_threshold: Minimum similarity to return (default 0.21)
        top_k: Maximum number of results to return
    
    Returns:
        List of dicts with file_id, file_path, similarity
    """
    # Load Index
    index, metadata = load_faiss_index()
    file_ids = metadata['file_ids']
    
    # Embed the text query
    query_embedding = embed_text(query_text)
    query_embedding = query_embedding.reshape(1, -1)
    
    # Handle dimension mismatch between CLIP (512) and FAISS index
    index_dim = index.d
    query_dim = query_embedding.shape[1]
    
    if query_dim != index_dim:
        print(f"Dimension mismatch: CLIP={query_dim}, FAISS={index_dim}")
        
        if query_dim > index_dim:
            # Truncate CLIP embedding to match FAISS index
            query_embedding = query_embedding[:, :index_dim]
        else:
            # Pad FAISS index with zeros (not ideal, but prevents crash)
            padding = np.zeros((1, index_dim - query_dim), dtype=np.float32)
            query_embedding = np.hstack([query_embedding, padding])
        
        print(f"Adjusted query dimension: {query_embedding.shape[1]}")
    
    # Search index
    distances, indices = index.search(query_embedding, top_k)
    
    # Load file paths
    result_file_ids = [file_ids[i] for i in indices[0] if i >= 0]
    path_dict = load_file_paths_from_db(result_file_ids)
    
    # Build results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        
        sim = float(distances[0, i])
        if sim < similarity_threshold:
            continue
        
        fid = file_ids[idx]
        results.append({
            'file_id': fid,
            'file_path': path_dict.get(fid, ""),
            'similarity': sim
        })
    
    # Sort by similarity descending
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results