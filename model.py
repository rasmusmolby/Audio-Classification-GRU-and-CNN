import torch
import torch.nn as nn

'''
Paper dident specify activation functions in cnn block.

Using MFCC with a 2:2 CNN and GRU.
Sigmoid at end for binary crossentropy classification with softmax. shit balls

Attention should be implemented in further iterations to increase accuracy (after every gru)


Remember to fix the transpose for the data extractor later on, as GRU wants a different tensor shape.


'''
class CNNGRU(nn.Module):
    def __init__(self, n_mfcc=39, c_cnn=32, n_classes = 3, gru_state=64):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(n_mfcc, c_cnn, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_cnn),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(c_cnn, c_cnn * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_cnn * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            
        )
        self.gru1 = nn.GRU(input_size=c_cnn, hidden_size=gru_state, batch_first=True)
        self.gru2 = nn.GRU(input_size=c_cnn, hidden_size=gru_state, batch_first=True) 
        self.fc = nn.Linear(in_features= 123, out_features=n_classes)





