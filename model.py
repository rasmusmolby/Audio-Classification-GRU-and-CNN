import torch
import torch.nn as nn

'''
Paper dident specify activation functions in cnn block.

Using MFCC with a 2:2 CNN and GRU.
No softmax cuz entropy so suck it



DONE Remember to fix the transpose for the data extractor later on, as GRU wants a different tensor shape.

mfcc outputs x lenght feature vector at each timestep. 

(batch,n_mfcc,time)

'''

# Returns weighted sum on each time frame.
# Basically just self attention module lmao
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        score = self.attention(x)
        weights = torch.softmax(score, dim=1)
        return (weights * x).sum(dim=1)


# Model itself
# 
# Hardcoded so no config management.
# TODO maybe add to config? idc
class CNNGRU(nn.Module):
    def __init__(self, n_mfcc=39, c_cnn=32, n_classes = 3, gru_state=64):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(n_mfcc, c_cnn, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_cnn),
            nn.LeakyReLU(), # I dont know why the original did not use leakyrelu. So i am using it now cuz fuck the law
            nn.MaxPool1d(2, ceil_mode=True),

        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(c_cnn, c_cnn * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_cnn * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
        )
        self.gru1 = nn.GRU(input_size=c_cnn * 2, hidden_size=gru_state, batch_first=True)
        self.attention2 = TemporalAttention(hidden_size=gru_state)
        self.attention1 = nn.MultiheadAttention(embed_dim=gru_state, num_heads = 4, batch_first=True)
        self.gru2 = nn.GRU(input_size=gru_state, hidden_size=gru_state, batch_first=True) 
        self.fc1 = nn.Linear(in_features = gru_state, out_features = gru_state * 2)
        self.fc2 = nn.Linear(in_features=gru_state * 2, out_features= 3)
        self.dropout = nn.Dropout(p = 0.5)
        # self.softmax = nn.Softmax(dim = 1)
        self.lrelu = nn.LeakyReLU()


    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.permute(0,2,1) # GRU expects time first
        x, _ = self.gru1(x)
        x_att, _ = self.attention1(x, x, x)
        x = x + x_att # Residual
        x, _ = self.gru2(x)
        x = self.attention2(x)
        x = self.lrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.softmax(x) # Cuz i use cross entropy
        return x


