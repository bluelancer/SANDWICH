import torch.nn as nn
# Define the simple model
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaselineMLP, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x