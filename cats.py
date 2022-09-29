"""
{CNN or perceptron} on {faces or dogs/cats} datasets
JJV for Deep Learning course, 2022
"""
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Nice progress bars


# DATA = 'catsdogs'
DATA = 'faces'
N_EPOCHS = 30 if DATA == 'faces' else 2
LEARNING_RATE = 0.01
BATCH_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TIME = datetime.now().strftime("%Hh%Mm%Ss")


def preprocess_data():
    '''
    Prepare datasets.
    Perform various operations (matrix rotation, normalization),
    then split into train and test datasets.
    Returns iterators over train and test.
    '''
    if DATA == 'catsdogs':
        data_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(
            root='dogscats/train', transform=data_transform)
        test_dataset = datasets.ImageFolder(
            root='dogscats/valid', transform=data_transform)
        # plt.imshow(train_dataset[0][0].permute(1, 2, 0).numpy())  # Display
        # plt.show()
        input_shape = (3, 224, 224)
        reduced_shape = (32, 27, 27)
    else:
        faces = fetch_lfw_people(min_faces_per_person=70, color=True)
        # plt.imshow(faces.images[0])  # Display one image
        # plt.show()
        X = torch.Tensor(faces.images).permute(0, 3, 1, 2)
        y = torch.LongTensor(faces.target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        input_shape = (3, 62, 47)
        reduced_shape = (32, 7, 5)

    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    return train_iter, test_iter, input_shape, reduced_shape


class NeuralNetwork(nn.Module):
    """
    CNN or perceptron (one fully connected layer).
    """
    def __init__(self, input_shape, reduced_shape, with_cnn=False):
        super().__init__()
        self.with_cnn = with_cnn
        self.input_shape = input_shape
        self.reduced_shape = reduced_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8)),
            # nn.Conv2d(32, 64, (3, 3)),
            # nn.ReLU(),
            # nn.MaxPool2d((4, 4)),
        )
        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=np.prod(self.reduced_shape) if with_cnn
                      else np.prod(self.input_shape), out_features=7),
        )

    def forward(self, x):
        if self.with_cnn:
            x = self.conv_layers(x)
        logits = self.fully_connected_layers(x)
        return logits

    def __str__(self):
        return 'cnn' if self.with_cnn else 'perceptron'


def train(dataloader, model, loss_function, optimizer, writer):
    model.train()  # Training mode
    losses = []
    accuracies = [0.]
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        writer.add_graph(model, inputs)  # Display graph in TensorBoard
        # writer.add_images('images', inputs)  # Display images in TensorBoard

        # Compute prediction error
        logits = model(inputs)
        loss = loss_function(logits, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(axis=1)
        accuracies.append(torch.sum(predictions == targets).item())

        losses.append(loss.item())
    return np.mean(losses), np.sum(accuracies) / len(dataloader.dataset)


def test(dataloader, model, loss_function):
    model.eval()  # Test mode
    accuracies = []
    with torch.no_grad():  # No training
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(inputs)
            predictions = logits.argmax(axis=1)
            accuracies.append(torch.sum(predictions == targets).item())
    return np.sum(accuracies) / len(dataloader.dataset)


if __name__ == '__main__':
    train_iter, test_iter, input_shape, reduced_shape = preprocess_data()

    for with_cnn in (False, True):
        model = NeuralNetwork(input_shape, reduced_shape, with_cnn).to(DEVICE)
        writer = SummaryWriter(log_dir=f'logs/fit/{model}-{TIME}')  # TBoard

        n_parameters = 0
        for name, parameter in model.named_parameters():
            print(name, parameter.numel())
            n_parameters += parameter.numel()
        print(f'Total number of parameters of {model}: {n_parameters}')

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(N_EPOCHS):
            print(f'=== Epoch {epoch} ===')
            train_loss, train_acc = train(train_iter, model, loss_function,
                                          optimizer, writer)
            print(f'Train loss: {train_loss:7f} / Train acc: {train_acc:.2f}')
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)

            test_acc = test(test_iter, model, loss_function)
            print(f'Test accuracy: {test_acc:.2f}')
            writer.add_scalar('Accuracy/test', test_acc, epoch)

        writer.close()
