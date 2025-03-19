"""
    This module is used to run inference.

    Author: Jesse Little
    Date: 1/18/2025
"""

#Imports
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import cohere


#Constants
db_name = 'kjv_db.index'
db_path = os.path.join('model', db_name)
metadata_name = 'metadata.json'
metadata_path = os.path.join('output', metadata_name)
retrieval_model = SentenceTransformer('all-mpnet-base-v2')

coherent_api_key = "zoZZMkj9MNPPqNGuujfvfDicYoDTjkkOYhb4nzcx"
co = cohere.Client(coherent_api_key)

#Functions
def load_db():
    """
    This function will read the index and metadata from the disk.
    """

    index = faiss.read_index(db_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return index, metadata


def get_query():
    """
    This function will get a query from the user.
    """

    query = input('Enter a query: ')
    return query

def get_cohere_response(context, user_question):
    """
    Sends a question to OpenAI with provided context.
    """
    try:
        prompt = f"""
        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """
        
        response = co.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7,
        )

        return response.generations[0].text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I couldn't process the request at the moment. Please try again later."