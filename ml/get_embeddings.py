import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import sent_tokenize

# Load the dataset from a local CSV file
df = pd.read_csv('search_dataset.csv')

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Clean the 'movie_info' column: convert to string and handle NaN values
df['movie_info'] = df['movie_info'].fillna('').astype(str)

# Create a new DataFrame to store individual sentences
sentences_df = pd.DataFrame(columns=['sentence_id', 'sentence', 'embedding'])

# Process each movie_info
for index, row in df.iterrows():
    rotten_id = row['rotten_tomatoes_link']
    movie_info = row['movie_info']

    # Split movie_info into sentences
    sentences = sent_tokenize(movie_info)

    # Process each sentence
    for sent_num, sentence in enumerate(sentences, 1):
        # Create a unique sentence_id
        sentence_id = f"{rotten_id}_{sent_num}"

        # Generate embedding for the sentence
        embedding = model.encode(sentence).tolist()

        # Add to the new DataFrame
        new_row = pd.DataFrame([{
            'sentence_id': sentence_id,
            'sentence': sentence,
            'embedding': embedding
        }])

        sentences_df = pd.concat([sentences_df, new_row], ignore_index=True)

# Display the first few rows of the new DataFrame
print(sentences_df.head())

# Save the new DataFrame to a CSV file
sentences_df.to_csv('rotten_tomatoes_sentences_with_embeddings.csv', index=False)
