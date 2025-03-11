import torch.nn as nn


# Define the simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.fcs = nn.Sequential(*[nn.Linear(256, 256) for _ in range(n_layers)])
        self.relu_fcs = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fcs(x)
        x = self.relu_fcs(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x