epochs = 15

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

from splitnn_net import SplitNN
from distribute_data import Distribute_MNIST

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('mnist', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

data_owners = ['client_1', 'client_2']
model_locations = ['client_1', 'client_2', 'server']
# Split each image and send one part to client_1, and other to client_2
distributed_trainloader = Distribute_MNIST(data_owners=data_owners, data_loader=trainloader)

input_size = [28 * 14, 28 * 14]
hidden_sizes = {"client_1": [32], "client_2": [32], "server": [32, 128]}

# create model segment for each worker
models = {
    "client_1": nn.Sequential(
        nn.Linear(input_size[0], hidden_sizes["client_1"][0]),
        nn.ReLU(),

    ),

    "client_2": nn.Sequential(
        nn.Linear(input_size[1], hidden_sizes["client_2"][0]),
        nn.ReLU(),

    ),

    "server": nn.Sequential(

        nn.Linear(hidden_sizes["server"][0], hidden_sizes["server"][1]),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1)

    ),

}

# Create optimisers for each segment and link to their segment
optimizers = [
    optim.SGD(models[location].parameters(), lr=0.01, )
    for location in model_locations
]

splitnn = SplitNN(models, optimizers, data_owners).to(device)


def train(x, target, splitNN):
    # 1) Zero our grads
    splitNN.zero_grads()

    # 2) Make a prediction
    pred = splitNN.forward(x)

    # 3) Figure out how much we missed by
    criterion = nn.NLLLoss()
    loss = criterion(pred, target.reshape(-1, 64)[0])

    # 4) Backprop the loss on the end layer
    loss.backward()

    # 5) Feed Gradients backward through the network


    # 6) Change the weights
    splitNN.step()

    return loss.detach().item()


def cal_acc(model, dataloader, dataset_name):
    correct = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            output = splitnn.forward(data_ptr)
            pred = output.max(1, keepdim=True)[1]
            h = label.data.view_as(pred)
            correct += pred.eq(label.data.view_as(pred)).sum()

    file = open("acc.txt", 'a+')

    file.write("{:.2f}%".format(100. * correct / (len(dataloader) * 64)))
    file.write("\n")
    file.flush()
    file.close()

    print("{}: Accuracy {}/{} ({:.2f}%)".format(dataset_name,
                                                correct,
                                                len(dataloader) * 64,
                                                100. * correct / (len(dataloader) * 64)))


if __name__ == "__main__":
    for i in range(epochs):
        running_loss = 0
        splitnn.train()
        for images, labels in distributed_trainloader:
            loss = train(images, labels, splitnn)
            running_loss += loss
        else:
            print("Epoch {} - Training loss: {}".format(i, running_loss / len(trainloader)))
            cal_acc(models, distributed_trainloader, "Train set")

    testset = datasets.MNIST('mnist', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    distributed_testloader = Distribute_MNIST(data_owners=data_owners, data_loader=testloader)

    # Accuracy on train and test sets

    cal_acc(models, distributed_testloader, "Test set")
