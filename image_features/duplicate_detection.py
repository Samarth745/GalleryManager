import sqlite3
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Set
import os

# Paths
DB_PATH = "databases/media_database.db"
FAISS_INDEX_PATH = "databases/faiss_index.bin"
FAISS_META_PATH = "databases/faiss_meta.pkl"

# Similarity threshold (97% = 0.97)
SIMILARITY_THRESHOLD = 0.97

def load_embeddings_from_db(conn: sqlite3.Connection) -> tuple:
    """
    Load embeddings directly from SQLite database.
    
    Returns:
        Tuple of (embeddings_array, file_ids, file_paths)
    """
    cursor = conn.execute("""
        SELECT m.file_id, m.file_path, e.embedding 
        FROM all_media_embeddings e
        JOIN all_media m ON e.file_id = m.file_id
    """)
    rows = cursor.fetchall()
    
    if not rows:
        raise ValueError("No embeddings found in database")
    
    embeddings = []
    file_ids = []
    file_paths = []
    
    for file_id, file_path, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        # Normalize for cosine similarity
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
        file_ids.append(file_id)
        file_paths.append(file_path)
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(file_ids)} embeddings from database")
    
    return embeddings_array, file_ids, file_paths

def load_embeddings_from_faiss() -> tuple:
    """
    Load embeddings from FAISS index.
    
    Returns:
        Tuple of (embeddings_array, file_ids, file_paths)
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Load metadata
    with open(FAISS_META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    file_ids = metadata['file_ids']
    
    # Get embeddings from index
    embeddings_array = index.reconstruct_n(0, index.ntotal)
    
    # Load file paths from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT file_id, file_path FROM all_media WHERE file_id IN ({})".format(
            ",".join("?" * len(file_ids))
        ),
        file_ids
    )
    path_dict = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    
    file_paths = [path_dict.get(fid, "") for fid in file_ids]
    
    print(f"Loaded {len(file_ids)} embeddings from FAISS index")
    
    return embeddings_array, file_ids, file_paths

class UnionFind:
    """Union-Find data structure for clustering."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_clusters(self) -> dict:
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters

def find_duplicates_faiss(
    embeddings: np.ndarray,
    file_ids: List[str],
    file_paths: List[str],
    similarity_threshold: float = 0.97,
    search_k: int = 50
) -> List[List[dict]]:
    """
    Find duplicate clusters using FAISS + Union-Find.
    
    This ensures that if A~B and B~C, all three end up in the same cluster
    (transitive clustering), even if A and C are not directly similar.
    
    Args:
        embeddings: Normalized embedding vectors (N x dim)
        file_ids: List of file IDs
        file_paths: List of file paths
        similarity_threshold: Minimum similarity to consider duplicate (default 0.97)
        search_k: Number of nearest neighbors to search (default 50)
    
    Returns:
        List of duplicate clusters, each containing dicts with file info
    """
    dim = embeddings.shape[1]
    n = len(embeddings)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Search for more neighbors to catch transitive duplicates
    search_k = min(search_k, n)
    distances, indices = index.search(embeddings, search_k)
    
    # Union-Find to group all related items together
    uf = UnionFind(n)
    
    # Union all pairs that meet similarity threshold
    for i in range(n):
        for j in range(1, search_k):  # Skip index 0 (self)
            idx = indices[i, j]
            sim = distances[i, j]
            
            if idx >= 0 and sim >= similarity_threshold:
                uf.union(i, idx)
    
    # Get all clusters
    raw_clusters = uf.get_clusters()
    
    # Build final clusters with file info
    duplicate_clusters = []
    
    for root, member_indices in raw_clusters.items():
        if len(member_indices) > 1:
            cluster = []
            for idx in member_indices:
                # Find similarity to cluster representative (first member)
                rep_idx = member_indices[0]
                sim = float(np.dot(embeddings[idx], embeddings[rep_idx]))
                
                cluster.append({
                    'file_id': file_ids[idx],
                    'file_path': file_paths[idx],
                    'similarity': sim
                })
            # Sort by similarity descending
            cluster.sort(key=lambda x: x['similarity'], reverse=True)
            duplicate_clusters.append(cluster)
    
    # Sort clusters by size (largest first)
    duplicate_clusters.sort(key=lambda c: len(c), reverse=True)
    
    return duplicate_clusters

def get_duplicates_clusters(DB_PATH = DB_PATH):
    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Load embeddings from database
    embeddings, file_ids, file_paths = load_embeddings_from_db(conn)

    # Find duplicate clusters
    duplicate_clusters = find_duplicates_faiss(
        embeddings,
        file_ids,
        file_paths,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    return duplicate_clusters