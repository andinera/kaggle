import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split



class CustomDataset(Dataset):
    def __init__(self, csv_file):
        train_data = pd.read_csv(csv_file)
        x_features = ["Pclass", "Sex", "Age", "SibSp"]
        y_features = ["Survived"]
        train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
        X = pd.get_dummies(train_data[x_features])
        y = pd.get_dummies(train_data[y_features])
        self.x_data = torch.from_numpy(X.values)
        self.y_data = torch.from_numpy(y.values)[:, 0]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
csv_file = "./train.csv"
dataset = CustomDataset(csv_file)
batch_size = 64

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))



# # Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


device = "cpu"

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        size = batch_size * 5 * 4
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(size, size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(size, 2, dtype=torch.float64)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



epochs = 300
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")




torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")