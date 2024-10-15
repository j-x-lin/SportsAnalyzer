from PIL import Image

import torch
from torchvision.transforms import v2
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sportsanalyzer_utils import max_frame_count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('Using', device)

PATH = './models/viewrecognizer.pt'


# Data augmentation and normalization for training
# Just normalization for validation
def get_view_recognizer_data_transforms(train=False):
    if train:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((64, 36)),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32),
            v2.Normalize((0.48016809 * 255, 0.52537006 * 255, 0.34176871 * 255), (0.02471182 * 255, 0.06024992 * 255, 0.03274162 * 255))
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((64, 36)),
            v2.ToDtype(torch.float32),
            v2.Normalize((0.48016809 * 255, 0.52537006 * 255, 0.34176871 * 255), (0.02471182 * 255, 0.06024992 * 255, 0.03274162 * 255))
        ])


class Imageset(data.Dataset):
    def __init__(self, isTrain):
        self.is_train = isTrain

        # load training data
        f = open("data/labels/view", "r")
        f.readline()

        data = []
        labels = []

        train_labels = []

        while True:
            line = f.readline()

            if not line:
                break

            tokens = line.split()

            frameNum = int(tokens[0])
            train_labels.append(frameNum)

            if isTrain:
                label = int(tokens[1])
                labels.append(np.int64(label))

        if isTrain:
            for i in range(max_frame_count() + 1):
                if i in train_labels:
                    data.append(i)

            self.data = torch.tensor(np.array(data)).to(device)
            self.labels = torch.tensor(np.array(labels)).to(device)
        else:
            for i in range(max_frame_count() + 1):
                if i not in train_labels:
                    data.append(i)

            self.data = torch.tensor(np.array(data)).to(device)

    def __getitem__(self, index):
        image = get_view_recognizer_data_transforms(train=True)(Image.open('data/frames/%d.jpg' % self.data[index]))

        if self.is_train:
            return image.to(device), self.labels[index].to(device)
        else:
            return torch.tensor(np.array([image])).to(device)

    def __len__(self):
        return len(self.data)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5).to(device)
        self.pool1 = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(10, 56, 7).to(device)
        self.pool2 = nn.MaxPool2d(2, 2).to(device)
        self.conv3 = nn.Conv2d(56, 144, 5).to(device)
        self.fc1 = nn.Linear(8 * 1 * 144, 36).to(device)
        self.fc2 = nn.Linear(36, 2).to(device)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


# returns train_set, test_set
def get_data():
    batch_size = 64

    # full_train_size = len(full_train)
    # train_size = int(full_train_size*0.8)
    # val_size = full_train_size - train_size

    # trainset, valset = data.random_split(full_train, [train_size, val_size])

    trainloader = data.DataLoader(Imageset(True), batch_size=batch_size, shuffle=True)

    # valloader = data.DataLoader(valset, batch_size=batch_size,
    #                                         shuffle=True)

    testloader = data.DataLoader(Imageset(False), batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def get_view_recognizer_model():
    net = Net()
    net.load_state_dict(torch.load(PATH))

    return net


# conv_lines_train = {}
# no_lines_train = {}
# one_lines_train = {}
# net_lines_train = {}

# conv_lines_val = {}
# no_lines_val = {}
# one_lines_val = {}
# net_lines_val = {}

def train_model(model):
    trainloader, testloader = get_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    acc_train = 0
    acc_val = 0

    epochs = 0

    acc_train_arr = np.zeros(0)
    acc_val_arr = np.zeros(0)

    loss_train_arr = np.zeros(0)
    loss_val_arr = np.zeros(0)

    while acc_train < 1:
        correct = 0
        total = 0

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            y = model.forward(inputs)
            for i in range(y.shape[0]):
                if torch.argmax(y[i]) == labels[i]:
                    correct += 1

            total += len(labels)

            # print statistics
            running_loss += loss.item()

        acc_train = correct / total

        acc_train_arr = np.append(acc_train_arr, acc_train)
        loss_train_arr = np.append(loss_train_arr, running_loss)

        print('-' * 10)
        print('Epoch', epochs)

        print('Train loss:', running_loss)

        print('Train accuracy:', acc_train)

        # correct = 0
        #
        # running_loss = 0.0

        # for i, data in enumerate(valloader, 0):
        #     # get the inputs; data is a list of [inputs, labels]
        #     inputs, labels = data
        #
        #     # forward
        #     outputs = model(inputs)
        #
        #     loss = criterion(outputs, labels)
        #
        #     y = model.forward(inputs)
        #     for i in range(y.shape[0]):
        #         if torch.argmax(y[i]) == labels[i]:
        #             correct += 1
        #
        #     # print statistics
        #     running_loss += loss.item()

        # acc_val = correct / 5000

        # acc_val_arr = np.append(acc_val_arr, acc_val)
        # loss_val_arr = np.append(loss_val_arr, running_loss)

        # print('Validation loss:', running_loss)
        #
        # print('Validation accuracy:', acc_val)

        epochs += 1

        print('-' * 10)
        print()

    print('Finished Training')

    torch.save(model.state_dict(), PATH)

    return acc_train_arr, acc_val_arr, loss_train_arr, loss_val_arr, epochs


# acc_train_arr, acc_val_arr, loss_train_arr, loss_val_arr, epochs = train_model(get_view_recognizer_model())
# net = Net()
# convNet = ConvNet(5, 200, 2)
# noHiddenNet = NoHiddenNet()
# oneHiddenNet = OneHiddenNet(1536)

# def test_model(model):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
#     acc = 0
#     correct = 0
#
#     running_loss = 0.0
#
#     for i, data in enumerate(testloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#
#         loss = criterion(outputs, labels)
#
#         y = model.forward(inputs)
#         for i in range(y.shape[0]):
#             if torch.argmax(y[i]) == labels[i]:
#                 correct += 1
#
#         # print statistics
#         running_loss += loss.item()
#
#     acc = correct / 10000
#
#     print('Test loss:', running_loss)
#     print('Test accuracy:', acc)
#
#     return acc, running_loss
#
# test_acc, test_loss = test_model(net)
# print(test_acc, test_loss)

# no_lines_train[0.005] = np.append(acc_no_arr, epochs)
# one_lines_train[1536] = np.append(acc_one_arr, epochs)
# conv_lines_train[200] = np.append(acc_conv_arr, epochs)

# no_lines_val[0.005] = np.append(acc_no_val_arr, epochs)
# one_lines_val[1536] = np.append(acc_one_val_arr, epochs)
# conv_lines_val[200] = np.append(acc_conv_val_arr, epochs)

# for hyperparam in no_lines_train:
#     arr = no_lines_train[hyperparam]
#     plt.plot(np.arange(arr[-1]), arr[: arr.size - 2], label='lr=' + str(hyperparam) + ' train')
#
# for hyperparam in no_lines_val:
#     arr = no_lines_val[hyperparam]
#     plt.plot(np.arange(arr[-1]), arr[: -1], label='lr=' + str(hyperparam) + ' val')
#
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.title('No Hidden Layers')
# plt.legend()
# plt.show()
#
# for hyperparam in one_lines_train:
#     arr = one_lines_train[hyperparam]
#     plt.plot(np.arange(arr[-1]), arr[: -2], label='size=' + str(hyperparam) + ' train')
#
# for hyperparam in one_lines_train:
#     arr = one_lines_val[hyperparam]
#     plt.plot(np.arange(arr[-1]), arr[: -1], label='size=' + str(hyperparam) + ' val')
#
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.title('One Hidden Layer')
# plt.legend()
# plt.show()
#
# for hyperparam in conv_lines_train:
#     arr = conv_lines_train[hyperparam]
#     plt.plot(np.arange(arr[-1]), arr[: -2], label='m=' + str(hyperparam) + ' train')
#
# for hyperparam in conv_lines_val:
#     arr = conv_lines_val[hyperparam]
#     plt.plot(np.arange(arr[-1]), arr[: -1], label='m=' + str(hyperparam) + ' val')
#
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.title('Convolutional network')
# plt.legend()
# plt.show()
