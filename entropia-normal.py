import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import math

def calculate_shannon_entropy(image):
    histogram = Counter(image.flatten())
    total_pixels = image.size
    entropy = -sum((count / total_pixels) * math.log2(count / total_pixels) for count in histogram.values())
    return entropy

def process_images_in_folder(folder_path):
    entropies = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".tif")):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('L')  
            entropy = calculate_shannon_entropy(np.array(image))
            entropies.append((filename, entropy))
    return entropies

def save_entropies_to_csv(entropies, output_csv_path):
    df = pd.DataFrame(entropies, columns=['filename', 'entropy'])
    df.to_csv(output_csv_path, index=False)

folder_path = 'ambro'
output_csv_path = 'amb-original-entropia.csv'
entropies = process_images_in_folder(folder_path)
save_entropies_to_csv(entropies, output_csv_path)
