from annoy import AnnoyIndex
import numpy as np
from image_processing.embeding_processor import enocode_query  # Assuming you have this function

# Global variable for Annoy index
annoy_index = None

def build_annoy_index(df, n_trees=10):
    vectors = list(df["vectors"])
    f = vectors[0].shape[0]
    index = AnnoyIndex(f, 'angular')
    for i, vec in enumerate(vectors):
        index.add_item(i, vec)
    index.build(n_trees)
    return index

def get_similar_vectors(query, df, index, similarity_threshold, max_neighbors=200):
    encoded_query = enocode_query(query, []).flatten()
    nn_indices, dists = index.get_nns_by_vector(encoded_query, max_neighbors, include_distances=True)
    cos_sims = 1 - np.square(dists) / 2
    filtered_results = [(idx, sim) for idx, sim in zip(nn_indices, cos_sims) if sim >= similarity_threshold]
    filtered_indices = [idx for idx, _ in filtered_results]
    filtered_scores = [sim for _, sim in filtered_results]

    result_df = df.loc[filtered_indices].copy()
    result_df['similarity_score'] = filtered_scores
    return result_df

def get_similar_images_dataframe(df, query = , threshold=0.23)
    annoy_index = build_annoy_index(df)
    return get_similar_vectors(query, df, annoy_index, similarity_threshold=threshold)
