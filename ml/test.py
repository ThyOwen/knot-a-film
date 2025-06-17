import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('Sources/Resources/rotten_tomatoes_movies.csv')

# Initialize the tokenizer and model
model_checkpoint = "apple/ane-distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, trust_remote_code=True, return_dict=False,
)

# Function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Print the shape of the output
    print(f"Output shape: {outputs[0].shape}")

    # Take the mean of the last hidden state
    embedding = outputs[1].mean(dim=1)

    print(f"Embedding shape: {embedding.shape}")
    return embedding.squeeze()

# Generate embeddings for all movie descriptions
embeddings = []
for description in tqdm(df['movie_info'][:100]):
    if isinstance(description, str):
        embedding = get_embedding(description)
        if embedding.dim() == 1:
            embeddings.append(embedding)
        else:
            print(f"Skipping embedding with unexpected shape: {embedding.shape}")

print("done")

# Convert embeddings list to tensor
embeddings_tensor = torch.stack(embeddings)
print(f"Embeddings tensor shape: {embeddings_tensor.shape}")

# Function to find similar movies
def find_similar_movies(query_description, top_k=5):
    query_embedding = get_embedding(query_description)
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
    similarities = cosine_similarity(query_embedding, embeddings_tensor)
    top_indices = similarities.argsort(descending=True)[0][:top_k]
    return df.iloc[top_indices]

# Example usage
query = "A group of superheroes must save the world from an alien invasion"
similar_movies = find_similar_movies(query)
print(similar_movies[['title', 'movie_info']])
