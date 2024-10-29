# Requirement 1
get_ipython().system('pip install nni')

# Requirement 2
get_ipython().system('pip install pytorch-lightning')

import torch.nn as nn
import nni
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear
import os
from pathlib import Path
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
import nni.nas.strategy as strategy

# Step 1: Define Model Space(Search Space + Basic Model)

# NOTE: Inherits nni module ModelSpace
class Net(ModelSpace):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        # Model Space parameter 1 : Convolutional Layer choice(NOTE the choices of two conv layers)
        self.conv2 = LayerChoice([
                           nn.Conv2d(32, 64, 3, 1),
                           nn.Conv2d(32, 64, 5, 1)
                       ], label='conv2')
        self.pool = nn.MaxPool2d(2, 2)
        # Model Space parameter 2 : Dropout Layer choice(NOTE the dropout param choices)
        self.drop = MutableDropout(nni.choice('dropout', [0.25, 0.5, 0.75]))
        # Model Space parameter 3 : Number of hidden units in FC layer
        feature = nni.choice('feature', [64, 128, 256])
        self.fc1 = MutableLinear(9216, feature)
        self.fc2 = MutableLinear(feature, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
print(model)

# Step 2: Define Model Evaluator

def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)

    for epoch in range(3):
        # Train
        train_epoch(model, device, train_loader, optimizer, epoch)
        # Test
        accuracy = test_epoch(model, device, test_loader)
        # Report intermediate accuracy
        nni.report_intermediate_result(accuracy)

    # Report final test accuracy
    nni.report_final_result(accuracy)

def evaluate_model_with_visualization(model):
    dummy_input = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(model, (dummy_input, ),
                      Path(os.getcwd() / 'model.onnx'))
    evaluate_model(model)

# Step 3: Create Evaluator

evaluator = FunctionalEvaluator(evaluate_model_with_visualization)

# Step 4: Create our NAS 'Experiment'

search_strategy = strategy.Random() # Other search strategies include GridSearch, DARTS(one-shot), ENAS(RL based controller)
exp = NasExperiment(model, evaluator, search_strategy)
exp.config.max_trial_number = 3 # Spawn 3 trials at most
exp.config.trial_concurrency = 1
exp.config.trial_gpu_number = 1 # Use GPU
exp.config.training_service.use_active_gpu = True

# Step 5: Run Experiment

get_ipython().system('nnictl stop --all')
exp.run(port=8081)

# Step 6: Export Top performing Models

for model_dict in exp.export_top_models(formatter='dict'):
    print('Top performing model :- ')
    print(model_dict)
