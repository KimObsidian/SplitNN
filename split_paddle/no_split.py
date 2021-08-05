import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from process_data import prepare_movielens_data
import numpy as np


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(160, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 3952)
        self.softmax5 = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax5(x)

        return x


model = Net()

watch_vec_size = 64
search_vec_size = 64
other_feat_size = 32
label_path = "/Users/lizhenyu/PycharmProjects/YoutubeDNN/label.csv"
user_watch, user_search, user_feat, user_labels = prepare_movielens_data(0, 32, watch_vec_size, search_vec_size,
                                                                         other_feat_size, 6040, label_path)
inputs = np.hstack((user_watch, user_search, user_feat))

test_inputs = inputs[5436:]

test_label = user_labels[5436:]
x_data_train = torch.FloatTensor(inputs)
y_data_train = torch.FloatTensor(user_labels)
x_data_test = torch.FloatTensor(test_inputs)
y_data_test = torch.FloatTensor(test_label)
deal_dataset = TensorDataset(x_data_train, y_data_train)
test_dataset = TensorDataset(x_data_test, y_data_test)
train_loader = DataLoader(dataset=deal_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01, )


def train(x, target, model):
    opt.zero_grad()
    pred = model(x)
    temp = target.reshape(-1, pred.shape[0])[0].long()
    loss = criterion(pred, temp)
    loss.backward()
    opt.step()
    return loss.detach().item()


def train_acc(train_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for input, label in train_loader:
            outputs = model(input)
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
    for i in range(100):
        running_loss = 0
        model.train()
        for images, labels in train_loader:
            loss = train(images, labels, model)
            running_loss += loss
        print("Epoch {} - Training loss:{}".format(i, running_loss / len(train_loader)))
        train_acc(train_loader)
        train_acc(test_loader)
