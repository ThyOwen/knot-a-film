from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import json
import glob

def download_poster_from_rotten_tomatoes_id(movie_id, output_dir="posters"):
    """
    Downloads the poster image for a movie from Rotten Tomatoes, given its ID.

    Args:
        movie_id (str): The movie ID (the part after `/m/` in the Rotten Tomatoes URL).
        output_dir (str, optional): The directory to save the downloaded image. Defaults to "posters".

    Returns:
        str: The filepath of the downloaded image, or None if the download failed.
    """
    url = f"https://www.rottentomatoes.com/{movie_id}"
    try:
    
        existing_files = glob.glob(os.path.join(output_dir, f"{movie_id[2:]}.*"))
        if existing_files:
            print(f"Image for movie ID {movie_id} already exists: {existing_files[0]}")
            return None
        else:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return download_poster_image(response.content, movie_id[2:])  # Pass the HTML content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HTML for movie ID {movie_id}: {e}")
        return None

def download_poster_image(html_content, movie_id, output_dir="posters"):
    """
    Searches HTML content for the movie poster image URL and downloads it.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Method 1: Look for a specific JSON-LD script
    script_tag = soup.find('script', {'type': 'application/ld+json'})
    if script_tag:
        try:
            data = json.loads(script_tag.string)
            image_url = data.get('image')
            if image_url:
                return download_image(image_url, movie_id, output_dir)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for movie ID {movie_id}")
    
    # Method 2: Look for a specific meta tag
    og_image = soup.find('meta', property='og:image')
    if og_image and og_image.get('content'):
        return download_image(og_image['content'], movie_id, output_dir)
    
    # Method 3: Search for img tags with specific attributes
    poster_img = soup.find('img', {'class': re.compile(r'poster|movie-poster')})
    if poster_img and poster_img.get('src'):
        return download_image(poster_img['src'], movie_id, output_dir)
    
    print(f"No poster image found for movie ID {movie_id}")
    return None

def download_image(image_url, movie_id, output_dir):
    """
    Downloads an image from a given URL if a file with the same movie_id doesn't already exist.
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if a file with the same movie_id already exists (regardless of extension)
        existing_files = glob.glob(os.path.join(output_dir, f"{movie_id}.*"))
        if existing_files:
            print(f"Image for movie ID {movie_id} already exists: {existing_files[0]}")
            return existing_files[0]
        
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        # Determine the filename
        file_extension = os.path.splitext(image_url.split('/')[-1])[-1]
        if not file_extension:
            file_extension = '.jpg'  # Default to .jpg if no extension is found
        filename = os.path.join(output_dir, f"{movie_id}{file_extension}")
        
        # Save the image
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Image downloaded to: {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image for movie ID {movie_id}: {e}")
        return None
       
def change_extensions_to_jpg(output_dir="posters"):
    # Check if the directory exists
    if not os.path.isdir(output_dir):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Get a list of all files in the directory
    files = os.listdir(output_dir)

    for filename in files:
        # Get the full file path
        file_path = os.path.join(output_dir, filename)

        # Check if it's a file (not a subdirectory)
        if os.path.isfile(file_path):
            # Split the filename and extension
            name, ext = os.path.splitext(filename)

            # Create the new filename with .jpg extension
            new_filename = f"{name}.jpg"
            new_file_path = os.path.join(output_dir, new_filename)

            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Example Usage:
if __name__ == '__main__':
    # 1. Load your Rotten Tomatoes dataset (replace 'rotten_tomatoes_data.csv' with your actual filename)
    try:
        df = pd.read_csv('rotten_tomatoes.csv')
    except FileNotFoundError:
        print("Error: rotten_tomatoes_data.csv not found.  Please put your csv file in the same directory")
        exit()

    # Assuming your DataFrame has a column named 'rotten_tomatoes_id' or similar
    #  Adjust the column name accordingly
    if 'rotten_tomatoes_link' not in df.columns:
        print("Error: No column named 'rotten_tomatoes_id' found in the csv. Check the column name and update the code.")
        exit()

    # 2. Iterate through the DataFrame and download posters
    for index, row in df.iterrows():
        movie_id = str(row['rotten_tomatoes_link'])  # Ensure movie_id is a string

        filepath = download_poster_from_rotten_tomatoes_id(movie_id)

        if filepath:
            print(f"Poster for movie ID {movie_id} downloaded successfully to {filepath}")
        else:
            print(f"Failed to download poster for movie ID {movie_id}")
    
    change_extensions_to_jpg()
