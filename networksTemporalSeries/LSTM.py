import torch
import torch.nn as nn

# https://www.youtube.com/watch?v=RHGiXPuo_pI

class LSTMnetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTMnetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        #NB: batch_first non si applica agli strati nascosti o alle celle di stato, ma solo a input e output!

        # Output size per layer LSTM, considerando bidirectional
        if self.bidirectional:
            output_size_lstm = hidden_size * 2
        else:
            output_size_lstm = hidden_size
        
        self.fc = nn.Linear(output_size_lstm, output_size)
        # self.sigmoid = nn.Sigmoid()  # Per la classificazione binaria


    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        if self.bidirectional:
            # Bidirectional LSTM: Extract forward hidden state
            out = self.fc(out[:, -1, 0:self.hidden_size])
        else:
            # Unidirectional LSTM: Use the original slicing
            out = self.fc(out[:, -1, :])

        #out = self.sigmoid(out)
        return out




