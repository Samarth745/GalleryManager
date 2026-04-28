import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def cluster_duplicate_images(df, similarity_threshold=0.95):
    # Stack and normalize embeddings
    embeds = torch.stack(df['vectors'].tolist())
    norm_embeds = embeds / embeds.norm(dim=1, keepdim=True)
    norm_embeds_np = norm_embeds.cpu().numpy()

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(norm_embeds_np)
    np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity

    # Build graph: nodes are images, edges for similarity above threshold
    G = nx.Graph()
    for i in range(len(df)):
        G.add_node(i)
    # Add edges between duplicates
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sim_matrix[i, j] > similarity_threshold:
                G.add_edge(i, j)
    # Extract connected components (clusters of duplicates)
    clusters = []
    for group in nx.connected_components(G):
        cluster_files = df.iloc[list(group)]['filepath'].tolist()
        if len(cluster_files) > 1:  # Ignore singletons
            clusters.append(cluster_files)
            mydict = {}
    m=0
    mydict={}
    for list_ in clusters:
        for k in list_:
            mydict[k]=m
        m+=1
    df["duplicated"]=df["filepath"].map(mydict).fillna(-1) 
    return pd.DataFrame(df[df["duplicated"]!=-1].groupby('duplicated')[["filepath", "bluriness"]].apply(lambda x:x.to_dict(orient='records'))).reset_index()


