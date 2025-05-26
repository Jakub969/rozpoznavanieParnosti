import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def generate_big_noise_dataset(num_images=10000, image_size=255, output_dir="dataset_big_noise"):
    os.makedirs(output_dir, exist_ok=True)

    images = []
    labels = []
    types = []

    print(f"Generujem {num_images} noise obrázkov veľkosti {image_size}x{image_size}...")

    for _ in tqdm(range(num_images)):
        # Vygeneruj náhodný binárny obraz (0 = čierny pixel, 1 = biely pixel)
        np_img = np.random.choice([0, 1], size=(image_size, image_size))

        # Konvertuj na obraz a ulož numpy
        img = Image.fromarray(np.uint8(255 * (1 - np_img)), mode='L')  # invertujeme pre správne zobrazenie
        img_np = (np.array(img) == 0).astype(np.uint8)  # čierne pixely ako 1

        num_black_pixels = np.sum(img_np)
        label = 0 if num_black_pixels % 2 == 0 else 1  # 0 = párny, 1 = nepárny

        images.append(img_np)
        labels.append(label)
        types.append("noise")

    # Konverzia a uloženie
    images = np.array(images)
    labels = np.array(labels)
    types = np.array(types)

    np.save(os.path.join(output_dir, "images.npy"), images)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "types.npy"), types)

    print(f"Dataset uložený do: {output_dir}/ (veľkosť: {image_size}x{image_size}, noise-only)")
