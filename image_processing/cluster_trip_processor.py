import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

def find_sessions(df, multiplier=2.0):
    def parse_datetime(s):
        try:
            return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S%z')

    df['date_taken'] = df['date_taken'].apply(parse_datetime)
    df = df.sort_values('date_taken')
    gaps = df['date_taken'].diff().dt.total_seconds() / 3600  # hours
    
    # Adaptive threshold: median gap × multiplier (robust to outliers)
    median_gap = gaps.median()
    threshold = median_gap * multiplier
    
    df['session_id'] = (gaps > threshold).cumsum()
    return df


def engineer_features(df):
    df['day_of_week'] = df['date_taken'].dt.dayofweek / 6.0
    
    # Burst density: photos per hour in a ±2h window
    df['burst_density'] = df.groupby('session_id')['date_taken'].transform(
        lambda s: len(s) / max((s.max() - s.min()).total_seconds() / 3600, 1)
    )
    return df


def build_similarity(df, alpha=0.6, beta=0.4, decay_hours=48):
    # Visual similarity (CLIP)
    embeddings = list(df["embedings"])
    timestamps = list(df["date_taken"])
    visual_sim = cosine_similarity(embeddings)
    
    # Temporal decay: exponential — photos close in time are more similar
    timestamps_numeric = np.array([(t - timestamps[0]).total_seconds() / 3600 for t in timestamps])
    time_hours = timestamps_numeric.reshape(-1, 1)
    time_diff = np.abs(time_hours - time_hours.T)
    temporal_sim = np.exp(-time_diff / decay_hours)
    t = alpha * visual_sim + beta * temporal_sim
    clusterer = hdbscan.HDBSCAN(
    min_cluster_size=8,       # minimum photos to form a trip
    min_samples=3,            # controls noise sensitivity
    metric='precomputed',     # use your fused distance matrix
    cluster_selection_method='eom'  # excess of mass: better for variable density
    )

    distance_matrix = 1 - t  # convert similarity → distance
    labels = clusterer.fit_predict(distance_matrix)
    
    return labels