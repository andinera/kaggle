import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, csv_file):
        train_data = pd.read_csv(csv_file)
        x_features = ["Pclass", "Sex", "Age", "SibSp"]
        id_features = ["PassengerId"]
        train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
        X = pd.get_dummies(train_data[x_features])
        id = pd.get_dummies(train_data[id_features])
        self.x_data = torch.from_numpy(X.values)
        self.id_data = torch.from_numpy(id.values)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.id_data[idx]
    
csv_file = "./test.csv"
dataset = CustomDataset(csv_file)
batch_size = 64


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
model.load_state_dict(torch.load("model.pth", weights_only=True))


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


model.eval()
with torch.no_grad():
    data_list = []
    for x, id in dataset:
        x = x.to(device)
        pred = model(x.unsqueeze(0))
        data_list.append([id.item(), pred[0].argmax(0).item()])

output = pd.DataFrame(data_list, columns=['PassengerId', 'Survived'])
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")



