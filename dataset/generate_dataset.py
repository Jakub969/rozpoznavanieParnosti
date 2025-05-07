import numpy as np
import os
from tqdm import tqdm

IMAGE_SIZE = 28
NUM_IMAGES = 5000
OUTPUT_DIR = ""

# Vytvorenie adresárov
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# Polia na uloženie dát a labelov
images = []
labels = []

for i in tqdm(range(NUM_IMAGES)):
    # Generovanie náhodného binárneho obrázka
    img = np.random.choice([0, 1], size=(IMAGE_SIZE, IMAGE_SIZE))

    # Spočítanie počtu čiernych pixelov (1 znamená čierny pixel)
    num_black_pixels = np.sum(img)

    # Určenie labelu: 0 = párny počet, 1 = nepárny počet
    label = 0 if num_black_pixels % 2 == 0 else 1

    images.append(img)
    labels.append(label)

# Konverzia na numpy array
images = np.array(images)
labels = np.array(labels)

# Uloženie datasetu
np.save(os.path.join(OUTPUT_DIR, "images.npy"), images)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

print(f"Dataset bol úspešne vygenerovaný! {NUM_IMAGES} obrázkov uložených v {OUTPUT_DIR}")
