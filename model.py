import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, dim):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(6, dim)
        self.fc = nn.Linear(dim, 1)
    
    def forward(self, x, hidden):
        outputs, hidden = self.lstm(x, hidden)
        outputs = F.relu(outputs)
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        x = self.fc(outputs)
        return x, hidden
