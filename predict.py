import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np


# Define command line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Parser of prediction script")
    parser.add_argument(
        "image_dir", type=str, help="Provide path to image. Mandatory argument"
    )
    parser.add_argument(
        "load_dir", type=str, help="Provide path to checkpoint. Mandatory argument"
    )
    parser.add_argument(
        "--top_k", type=int, default=1, help="Top K most likely classes. Optional"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Mapping of categories to real names. JSON file name to be provided. Optional",
    )
    parser.add_argument(
        "--GPU", action="store_true", help="Option to use GPU. Optional"
    )
    return parser.parse_args()


# Load a checkpoint and rebuild the model
def load_model(file_path):
    checkpoint = torch.load(file_path)
    if checkpoint["architecture"] == "alexnet":
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    for param in model.parameters():
        param.requires_grad = False
    return model


# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    im = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    im = transform(im).float()
    return im


# Predict the class of an image using a trained deep learning model
def predict(image_path, model, top_k, device):
    model.to(device)
    model.eval()

    # Process image and add batch dimension
    image = process_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.forward(image)
    output_prob = torch.exp(output)  # Convert to probabilities

    probs, indices = output_prob.topk(top_k)
    probs = probs.cpu().numpy().flatten()  # Convert to numpy array
    indices = indices.cpu().numpy().flatten()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    return probs, classes


# Main function
def main():
    args = get_input_args()

    device = torch.device("cuda" if args.GPU and torch.cuda.is_available() else "cpu")

    model = load_model(args.load_dir)

    cat_to_name = load_cat_names(args.category_names)

    probs, classes = predict(args.image_dir, model, args.top_k, device)

    class_names = [cat_to_name[str(cls)] for cls in classes]

    for i in range(args.top_k):
        print(
            f"Number: {i+1}/{args.top_k}.. "
            f"Class name: {class_names[i]}.. "
            f"Probability: {probs[i]*100:.3f}% "
        )


# Load category names
def load_cat_names(category_names_path):
    with open(category_names_path, "r") as f:
        cat_to_name = json.load(f)
    return cat_to_name


if __name__ == "__main__":
    main()
