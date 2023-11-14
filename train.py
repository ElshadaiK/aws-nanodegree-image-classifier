import argparse
import json
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for neural network")
    parser.add_argument("data_dir", type=str, help="Directory for the data")
    parser.add_argument(
        "--save_dir", type=str, help="Directory to save checkpoints", default="./"
    )
    parser.add_argument(
        "--arch",
        type=str,
        help='Model architecture: "vgg13" or "alexnet"',
        default="alexnet",
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.001)
    parser.add_argument(
        "--hidden_units", type=int, help="Hidden units for classifier", default=2048
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=7)
    parser.add_argument(
        "--GPU", action="store_true", help="Use GPU for training if available"
    )
    return parser.parse_args()


# Data loading
def load_data(data_dir):
    # Define transforms
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    valid_test_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
    valid_data = datasets.ImageFolder(
        data_dir + "/valid", transform=valid_test_transforms
    )
    test_data = datasets.ImageFolder(
        data_dir + "/test", transform=valid_test_transforms
    )
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return trainloader, validloader, testloader


# Build the model
def build_model(arch, hidden_units):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_features = 25088
    else:
        model = models.alexnet(pretrained=True)
        input_features = 9216
        arch = "alexnet"  # Ensuring default to 'alexnet' if not 'vgg13'

    # Freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(input_features, 4096)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(p=0.3)),
                ("fc2", nn.Linear(4096, hidden_units)),
                ("relu2", nn.ReLU()),
                ("dropout2", nn.Dropout(p=0.3)),
                ("fc3", nn.Linear(hidden_units, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier
    return model, arch


# Validate the model
def validate(model, dataloader, criterion, device):
    model.eval()
    model.to(device)
    valid_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            probabilities = torch.exp(outputs)
            equality = labels.data == probabilities.max(dim=1)[1]
            accuracy += equality.type(torch.FloatTensor).mean().item()
    return valid_loss, accuracy


# Main function
def main():
    args = parse_args()
    trainloader, validloader, testloader = load_data(args.data_dir)
    model, arch = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if args.GPU and torch.cuda.is_available() else "cpu")

    model.to(device)

    for e in range(args.epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        valid_loss, accuracy = validate(model, validloader, criterion, device)
        print(
            f"Epoch {e+1}/{args.epochs}.. "
            f"Training loss: {running_loss/len(trainloader):.3f}.. "
            f"Validation loss: {valid_loss/len(validloader):.3f}.. "
            f"Validation accuracy: {accuracy/len(validloader):.3f}"
        )

    # Do testing on test data here if needed

    # Save the checkpoint
    model.class_to_idx = trainloader.dataset.class_to_idx
    checkpoint = {
        "architecture": arch,
        "classifier": model.classifier,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "hidden_units": args.hidden_units,
    }
    torch.save(checkpoint, f"{args.save_dir}checkpoint.pth")

    print("Training complete, checkpoint saved.")


if __name__ == "__main__":
    main()
