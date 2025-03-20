# KJV-RAG: King James Version Retrieval-Augmented Generation

This repository provides tools for working with the King James Version (KJV) of the Bible, including data cleaning, database building, embedding generation, and visualization. It is designed to support retrieval-augmented generation (RAG) tasks.

## Features

- **Data Cleaning**: Preprocess raw KJV text to remove unwanted characters and prepare it for embedding generation.
- **Database Building**: Create a FAISS-based database for efficient similarity search using dense embeddings.
- **Embedding Visualization**: Visualize Bible verse embeddings in 3D using PCA and t-SNE.
- **Interactive Application**: A Streamlit-based app for querying the database and retrieving relevant Bible verses.

## Project Structure

### Key Files and Directories

- **`data/`**: Contains raw and cleaned versions of the KJV text.
- **`model/`**: Stores the FAISS index and embeddings.
- **`output/`**: Contains metadata and other generated outputs.
- **`src/`**: Contains Python scripts for various tasks:
  - `data_cleaning.py`: Cleans and preprocesses the raw KJV text.
  - `building_db.py`: Builds the FAISS database and generates embeddings.
  - `visualize_db.py`: Visualizes embeddings in 3D.
  - `app.py`: Streamlit app for querying the database.
  - `inference.py`: Handles inference and retrieval tasks.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/kjv-rag.git
   cd kjv-rag

   ```

2. Install the required dependencies:
   ```python
   pip install -r requirements.txt
   ```

## Usage

### Data Cleaning

Run `data_cleaning.py` script to preprocess the raw KJV text.

### Building the Database

Run `building_db.py` to build the FAISS database and generate embeddings.

### Vizualize Embeddings

Run `visualize_db.py` to visualize the embeddings in 3D using PCA and t-SNE.

### Running the application

Launch the Streamlit app to query the database `streamlit run app.py`

## Outputs

- **Cleaned Data**: Stored in `data/kjv_cleaned.csv`
- **Database Index**: Stored in `model/kjv_db.index`
- **Embeddings**: Stored in `model/kjv_embeddings.npy` and visualized in PCA/t-SNE formats.

## Contributing

Contributions are welcome! Feel free to open and issue or submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Aknowledgements

- The Pioneer Bible Project for providing the raw data
- Tools and libraries used including FAISS, SentenceTransformers, and Streamlit.
