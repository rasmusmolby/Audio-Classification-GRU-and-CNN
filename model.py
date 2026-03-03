import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    def __init__(self, n_mfcc=39, c_cnn=64, n_classes = 3, gru_state=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, c_cnn, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_cnn),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            
        )
        self.gru = nn.GRU(input_size=c_cnn, hidden_size=gru_state, batch_first=True) 
        self.fc = nn.Linear()





def convloop1d(data, config):
    nn.Conv1d(kernel_size=3, stride=1, in_channels=(39, 1), out_channels=(39, 32))



    return null
