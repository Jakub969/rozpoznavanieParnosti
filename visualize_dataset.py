import numpy as np
import matplotlib.pyplot as plt

def visualize_random_images(dataset_path="dataset_shapes", num_images=9):
    # Načítanie datasetu
    images = np.load(f"{dataset_path}/images.npy")
    labels = np.load(f"{dataset_path}/labels.npy")

    # Vyber náhodné indexy
    indices = np.random.choice(len(images), num_images, replace=False)

    # Vykreslenie
    plt.figure(figsize=(10, 10))

    for i, idx in enumerate(indices):
        img = images[idx]
        label = labels[idx]

        plt.subplot(3, 3, i+1)  # 3x3 grid
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Párnosť: {'Párny' if label == 0 else 'Nepárny'}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_random_images()
