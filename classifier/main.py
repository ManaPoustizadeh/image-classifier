import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from classifier.cnn import Net

class ClassificationConfig:
    def __init__(self):
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size: int = 4

class Classifier:
    def __init__(self):
        self.config = ClassificationConfig()
        self.load_data(self, self.config.transform, self.config.batch_size)
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
    
    def load_data(self, transform, batch_size: int):

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')