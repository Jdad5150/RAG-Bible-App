import faiss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import plotly.express as px
import pandas as pd

# Load the saved FAISS index and embeddings
def load_index_and_data():
    index = faiss.read_index('model/kjv_db.index')  # Load FAISS index
    embeddings = np.load('model/kjv_embeddings.npy')  # Load the original embeddings
    embeddings_3d = np.load('model/kjv_embeddings_t-SNE.npy')  # Load the t-SNE embeddings
    metadata = json.load(open('output/metadata.json'))  # Load metadata
    return index, embeddings, embeddings_3d, metadata

# Visualize the 3D embedding space using t-SNE
def visualize_embeddings(embeddings_3d):
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        title='3D Visualization of Bible Verse Embeddings',
        labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'}
    )
    fig.show()

# Optionally, you can visualize only a subset or focus on clustering
def plot_clustered_embeddings(embeddings, metadata):
    # Example: Plot some clusters by book name
    # You might need to encode your cluster labels based on your approach

    book_names = [entry['Book Name'] for entry in metadata]
    unique_books = list(set(book_names))
    num_books = (len(unique_books))

    colors = plt.colormaps['tab20'].resampled(num_books)  # Use a colormap
    color_map = {book: colors(i) for i, book in enumerate(unique_books)}


    df = pd.DataFrame(embeddings_3d, columns=["Dimension 1", "Dimension 2", "Dimension 3"])
    df['Book Name'] = book_names    

    # Create a Plotly 3D scatter plot with color by book name
    fig = px.scatter_3d(df, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='Book Name',
                        title="Clustered 3D Visualization of Bible Verse Embeddings",
                        labels={'Dimension 1': 'X', 'Dimension 2': 'Y', 'Dimension 3': 'Z'},
                        hover_data=['Book Name'],
                        width=1600,
                        height=1000,)

    # Show the plot
    fig.show()

# Main script execution
if __name__ == "__main__":
    index, embeddings, embeddings_3d, metadata = load_index_and_data()

    # Visualize embeddings (basic 3D plot)
    visualize_embeddings(embeddings_3d)

    # Optionally, visualize by clusters (e.g., by book name)
    plot_clustered_embeddings(embeddings_3d, metadata)
