import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

from torch import nn, optim

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.LeakyReLU()
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.lsm(x)
        return(x)

model = NN()
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.005)

epochs = 10

for e in range(epochs):
    train_loss = 0
    for image, target in trainloader:
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    else:
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for image, target in testloader:
                output = model(image)
                prob = torch.exp(output)
                guess = prob.topk(1)[1].view(-1)
                equals = guess == target
                accuracy = equals.type(torch.FloatTensor).mean()
                test_loss += loss_fn(output, target).item()
        model.train()
        print(f'Epoch {e+1}/{epochs}  |  Accuracy: {accuracy * 100}%)  |  train loss: {train_loss}  |  test less: {test_loss}')


checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [256, 64],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
