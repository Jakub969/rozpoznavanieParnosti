import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
import random

def generate_dataset(num_images=5000, output_dir="dataset_shapes"):

    # Nastavenia
    IMAGE_SIZE = 28

    # Typy obrázkov
    TYPES = ['noise', 'square', 'circle', 'line']

    # Vytvorenie adresárov
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Polia na uloženie dát a labelov
    images = []
    labels = []
    types = []

    for i in tqdm(range(num_images)):
        img_type = random.choice(TYPES)

        # Vytvor biely obrázok
        img = Image.new('1', (IMAGE_SIZE, IMAGE_SIZE), color=1)
        draw = ImageDraw.Draw(img)

        if img_type == 'noise':
            np_img = np.random.choice([0, 1], size=(IMAGE_SIZE, IMAGE_SIZE))
            img = Image.fromarray(np.uint8(255 * (1 - np_img)), mode='L')
        elif img_type == 'square':
            size = random.randint(5, 15)
            x0 = random.randint(0, IMAGE_SIZE - size)
            y0 = random.randint(0, IMAGE_SIZE - size)
            draw.rectangle([x0, y0, x0 + size, y0 + size], fill=0)
        elif img_type == 'circle':
            size = random.randint(5, 15)
            x0 = random.randint(0, IMAGE_SIZE - size)
            y0 = random.randint(0, IMAGE_SIZE - size)
            draw.ellipse([x0, y0, x0 + size, y0 + size], fill=0)
        elif img_type == 'line':
            x0 = random.randint(0, IMAGE_SIZE)
            y0 = random.randint(0, IMAGE_SIZE)
            x1 = random.randint(0, IMAGE_SIZE)
            y1 = random.randint(0, IMAGE_SIZE)
            draw.line([x0, y0, x1, y1], fill=0, width=random.randint(1, 3))

        img_np = np.array(img)
        img_np = (img_np == 0).astype(np.uint8)

        num_black_pixels = np.sum(img_np)
        label = 0 if num_black_pixels % 2 == 0 else 1

        images.append(img_np)
        labels.append(label)
        types.append(img_type)

    # Konverzia na numpy array
    images = np.array(images)
    labels = np.array(labels)

    # Uloženie datasetu
    np.save(os.path.join(output_dir, "images.npy"), images)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "types.npy"), np.array(types))

    print(f"Dataset s tvarmi bol úspešne vygenerovaný! {num_images} obrázkov uložených v {output_dir}")
