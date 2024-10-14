# Import packages
import torch.nn as nn


# Class
class RNNetwork(nn.Module):

    def __init__(self, num_classes=2):
        super(RNNetwork, self).__init__()

        # Define attributes
        self.num_classes = num_classes
        if num_classes == 2:
            self.in_channels = 75
            self.hidden_size = 32
            self.num_layers = 1
            self.output_neurons = 1
        else:
            # TODO
            self.in_channels = 75
            self.hidden_size = 32
            self.num_layers = 1
            self.output_neurons = 1
            self.output_neurons = num_classes

        # Layers
        self.rnn = nn.RNN(input_size=self.in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_neurons)
        if self.output_neurons == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1)


    def forward(self, x):
        try:
            out, _ = self.rnn(x)
        except AttributeError:
            # Handle previous versions of the LSTMNetwork class (no rnn attribute)
            out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out