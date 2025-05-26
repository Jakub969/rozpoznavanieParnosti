import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def generate_noise_only_dataset(num_images=10000, image_size=28, output_dir="dataset_noise_28x28"):
    os.makedirs(output_dir, exist_ok=True)

    images = []
    labels = []
    types = []

    print(f"Generujem {num_images} noise obrázkov veľkosti {image_size}x{image_size}...")

    for _ in tqdm(range(num_images)):
        # Náhodný binárny obraz (0 = čierny pixel, 1 = biely pixel)
        np_img = np.random.choice([0, 1], size=(image_size, image_size))

        # Konverzia a inverzia farieb (0 = čierny pixel)
        img = Image.fromarray(np.uint8(255 * (1 - np_img)), mode='L')
        img_np = (np.array(img) == 0).astype(np.uint8)  # 1 = čierny pixel

        # Spočítanie čiernych pixelov
        num_black_pixels = np.sum(img_np)
        label = 0 if num_black_pixels % 2 == 0 else 1

        images.append(img_np)
        labels.append(label)
        types.append("noise")

    # Uloženie ako numpy polia
    images = np.array(images)
    labels = np.array(labels)
    types = np.array(types)

    np.save(os.path.join(output_dir, "images.npy"), images)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "types.npy"), types)

    print(f"Dataset úspešne vygenerovaný: {output_dir}/ ({num_images} obrázkov, typ: noise)")
