from dataset.generate_big_noise_dataset import generate_big_noise_dataset
from dataset.generate_dataset_with_shapes import generate_dataset
from dataset.generate_noise_only_dataset import generate_noise_only_dataset
from model.big_input_shape_cnn import build_big_input_cnn
from model.cnn_model import build_model
from train.evaluate_crossdomain import evaluate_shapes_model_on_noise
from train.train_model import train_and_evaluate
from model.fcnn_model import build_fcnn_model

def main():
    #print("Generujem dataset...")
    #generate_dataset(num_images=10000, output_dir="dataset_shapes")
    #generate_big_noise_dataset(num_images=5000, image_size=255, output_dir="dataset_big_noise")
    #generate_noise_only_dataset(num_images=10000, image_size=28, output_dir="dataset_noise_28x28")

    print("Budujem model...")
    #model = build_model(input_shape=(28, 28, 1))
    #model = build_big_input_cnn(input_shape=(255, 255, 1))
    model = build_fcnn_model(input_shape=(28, 28, 1))

    print("Trénujem model...")
    train_and_evaluate(model, dataset_path="dataset_shapes", data_subset="shapes")
    #train_and_evaluate(model, dataset_path="dataset_noise_28x28", data_subset="all")
    #train_and_evaluate(model, dataset_path="dataset_big_noise", data_subset="noise")

if __name__ == "__main__":
    main()
    #evaluate_shapes_model_on_noise()
