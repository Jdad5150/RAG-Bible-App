"""
    This module is used to build the database for the RAG Model.
    The database is built using FAISS, a library for efficient similarity search and clustering of dense vectors.
    The database is built using the embeddings of the text data.

    Author: Jesse Little
    Date: 1/18/2025
"""

#Imports
import faiss
import numpy as np
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px

#Constants
file_name = 'kjv_cleaned.csv'
file_path = os.path.join('data', file_name)
embedding_name = 'kjv_embeddings'
embedding_path = os.path.join('model', embedding_name)
db_name = 'kjv_db.index'
db_path = os.path.join('model', db_name)

def build_db():
    
    df = pd.read_csv(file_path)
    print('Data Loaded Successfully...')
    print(df.head(5))

    verses = df['Text'].values
    metadata = df[['Book Name', 'Chapter', 'Verse']].to_dict(orient='records')

    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(verses, show_progress_bar=True)
    print('Embeddings Generated Successfully...')

    embeddings = np.array(embeddings).astype('float32')

    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=300)
    embeddings_3d = tsne.fit_transform(embeddings_pca)


    np.save('model/kjv_embeddings.npy', embeddings)  # Save original embeddings
    np.save('model/kjv_embeddings_pca.npy', embeddings_pca)  # Save PCA-reduced embeddings
    np.save('model/kvj_embeddings_t-SNE.npy', embeddings_3d)  # Save t-SNE embeddings


    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, db_path)

    with open('output/metadata.json', 'w') as f:
        json.dump(metadata, f)




if __name__ == '__main__':
    build_db()