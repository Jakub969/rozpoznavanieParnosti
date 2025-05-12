from dataset.generate_dataset_with_shapes import generate_dataset
from model.cnn_model import build_model
from train.train_model import train_and_evaluate
from model.fcnn_model import build_fcnn_model

def main():
    print("Generujem dataset...")
    generate_dataset(num_images=10000, output_dir="dataset_shapes")

    print("Budujem model...")
    model = build_model(input_shape=(28, 28, 1))
    #model = build_fcnn_model(input_shape=(28, 28, 1))

    print("Trénujem model...")
    train_and_evaluate(model, dataset_path="dataset_shapes", data_subset="noise")

if __name__ == "__main__":
    main()
