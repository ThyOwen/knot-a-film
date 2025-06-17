from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def extract_key_colors(image_path, num_colors=9):
    # Open the image
    img = Image.open(image_path)
    
    # Convert image to RGB mode if it's not already
    img = img.convert('RGB')
    
    # Reshape the image data into a 2D array of pixels
    pixels = np.array(img).reshape(-1, 3)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_.astype(int)
    
    return colors


if __name__ == '__main__':
    # 1. Load your Rotten Tomatoes dataset
    try:
        df = pd.read_csv('rotten_tomatoes.csv')
    except FileNotFoundError:
        print("Error: rotten_tomatoes.csv not found. Please put your csv file in the same directory")
        exit()

    if 'rotten_tomatoes_link' not in df.columns:
        print("Error: No column named 'rotten_tomatoes_link' found in the csv. Check the column name and update the code.")
        exit()

    # Initialize the new column for hex colors
    df['hex_colors'] = None

    # 2. Iterate through the DataFrame and extract colors
    for index, row in df.iterrows():
        movie_id = str(row['rotten_tomatoes_link'])
        movie_poster_path = "posters/" + movie_id[2:] + ".jpg"

        if os.path.exists(movie_poster_path):
            print("Poster : {}".format(row["movie_title"]))
            colors = extract_key_colors(movie_poster_path)
            hex_colors = [rgb_to_hex(color[0], color[1], color[2]) for color in colors]
            df.at[index, 'hex_colors'] = str(hex_colors)  # Store as string representation of list
            print(str(hex_colors))
        else:
            print(f"Warning: Poster not found for movie ID {movie_id}")

    # 3. Save the updated DataFrame
    df.to_csv('rotten_tomatoes_with_colors.csv', index=False)
    print("Updated CSV saved as 'rotten_tomatoes_with_colors.csv'")
