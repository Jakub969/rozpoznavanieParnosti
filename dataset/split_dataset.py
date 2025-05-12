import numpy as np

class DatasetSplitter:
    def __init__(self, dataset_path):
        self.images = np.load(f"{dataset_path}/images.npy")
        self.labels = np.load(f"{dataset_path}/labels.npy")
        self.types = np.load(f"{dataset_path}/types.npy")

    def split_by_type(self):
        noise_indices = np.where(self.types == 'noise')[0]
        shape_indices = np.where(self.types != 'noise')[0]

        noise_data = {
            'images': self.images[noise_indices],
            'labels': self.labels[noise_indices]
        }

        shape_data = {
            'images': self.images[shape_indices],
            'labels': self.labels[shape_indices]
        }

        return noise_data, shape_data
