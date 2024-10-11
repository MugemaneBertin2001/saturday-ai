import torch.nn as nn
import torch.nn.functional as F

class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        # Define layers as per the saved model architecture
        self.fc1 = nn.Linear(in_features=3029, out_features=2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=1024, out_features=3200)
        
        # Additional layers (Conv1D and BatchNorm1D)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Fully connected layers after convolution
        self.fc4 = nn.Linear(in_features=1536, out_features=256)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=256, out_features=64)
        
        # Output layer (3 classes for sentiment prediction)
        self.output = nn.Linear(in_features=64, out_features=3)

    def forward(self, x):
        # Assuming the input goes through a fully connected path first
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))

        # Assuming the model has Conv1D operations
        x = x.unsqueeze(1)  # Add channel dimension if needed
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        # Flattening or reshaping for further fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        x = self.dropout3(x)
        x = F.relu(self.fc5(x))
        
        # Output prediction
        x = self.output(x)
        
        return x
