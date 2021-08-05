from torch import nn, optim
import torch
from process_data import prepare_movielens_data, prepare_movielens_test_data
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from distribute_data import Distribute_Youtube
from splitnn_net import SplitNN

input_size = [64, 96]
hidden_sizes = {"client_1": [128], "client_2": [128], "server": [128, 128]}

models = {
    "client_1": nn.Sequential(
        nn.Linear(input_size[0], 64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
    ),
    "client_2": nn.Sequential(
        nn.Linear(input_size[1], 96),
        nn.ReLU(),
        nn.Linear(96,64),
        nn.ReLU(),
    ),
    "server": nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 3952),
        nn.LogSoftmax(dim=1)
    )
}

watch_vec_size = 64
search_vec_size = 64
other_feat_size = 32
label_path = "/Users/lizhenyu/PycharmProjects/YoutubeDNN/label.csv"
user_watch, user_search, user_feat, user_labels = prepare_movielens_data(0, 32, watch_vec_size, search_vec_size,
                                                                         other_feat_size, 6040, label_path)
inputs = np.hstack((user_watch, user_search, user_feat))
x_data = torch.FloatTensor(inputs)
y_data = torch.FloatTensor(user_labels)

deal_dataset = TensorDataset(x_data, y_data)
train_size = int(0.91 * len(deal_dataset))
test_size = len(deal_dataset) - train_size
train_data_set, test_data_set = torch.utils.data.random_split(deal_dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_data_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data_set, batch_size=32, shuffle=True)

data_owners = ['client_1', 'client_2']
model_locations = ['client_1', 'client_2', 'server']
distributed_trainloader = Distribute_Youtube(data_owners=data_owners, data_loader=train_loader)
distributed_testloader = Distribute_Youtube(data_owners=data_owners, data_loader=test_loader)
optimizers = [optim.Adam(models[location].parameters(), lr=0.01, ) for location in model_locations]
splitnn = SplitNN(models, optimizers, data_owners)


def train(x, target, splitNN):
    splitNN.zero_grads()
    pred = splitNN.forward(x)
    criterion = nn.CrossEntropyLoss()
    temp = target.reshape(-1, pred.shape[0])[0].long()
    loss = criterion(pred, temp)
    loss.backward()
    splitNN.step()
    return loss.detach().item()


def train_acc(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs = splitnn.forward(data_ptr)
            value, indicies = torch.topk(outputs, 10, dim=1)
            total += label.size(0)
            correct += topK(label, indicies)
    print("Accuracy {:.2f}%".format(100 * correct / total))
    file = open("res.txt", 'a+')
    file.write("{:.2f}%\n".format(100. * correct / total))
    file.flush()
    file.close()


def topK(labels, indicy):
    upper = labels.size(0)
    labels = labels.numpy()
    indicy = indicy.numpy()
    hit = 0
    for i in range(upper):
        for h in range(10):
            if indicy[i][h] == labels[i][0]:
                hit += 1
                break
    return hit


if __name__ == "__main__":
    for i in range(25):
        running_loss = 0
        splitnn.train()
        for images, labels in distributed_trainloader:
            loss = train(images, labels, splitnn)
            running_loss += loss
        print("Epoch {} - Training loss:{}".format(i, running_loss / len(train_loader)))
        train_acc(distributed_trainloader)
        train_acc(distributed_testloader)
