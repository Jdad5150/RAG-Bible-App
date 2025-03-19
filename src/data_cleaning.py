"""
    This module is used to clean the data, and prepare it for the FAISS database.
    Author: Jesse Little
    Date: 1/18/2025
"""

import pandas as pd
import numpy as np
import os
import csv

def clean_data():

    """
    This function is used to clean the data, and prepare it for the FAISS database.
    """

    df = pd.read_csv('data/kjv_raw.csv')
    
    print('Data loaded successfully.')

    print('Unwanted ASCII character at the beginning of the some verses, that states when a new paragraph begins.')
    print('Example of data:')
    example_str = "Â¶ In the beginning God created the heaven and the earth."
    print(repr(example_str))
    print('Removing unwanted ASCII character from example string:')
    cleaned_str = example_str.replace('Â¶ ', '')
    print(repr(cleaned_str))
    print('Removing unwanted ASCII character from all verses.....')

    with open('data/kjv_raw.csv', mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        cleaned_rows = []

        for row in reader:
            row[5] = row[5].replace('¶ ', '').strip()
            cleaned_rows.append(row)

    print('Data cleaned successfully.')

    with open('data/kjv_cleaned.csv', mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)

    df = pd.read_csv('data/kjv_cleaned.csv')
    print(f'Data saved successfully, see first 5 rows below:\n{df.head(5)}')


if __name__ == '__main__':
    clean_data()