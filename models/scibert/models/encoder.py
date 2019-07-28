import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, 128, padding_idx=1)
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)

    def forward(self, input, hidden=None):
        # input = torch.LongTensor(input)  # batch * 1
        input = self.embedding(input)  # batch * embedding
        output, hidden = self.lstm(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
