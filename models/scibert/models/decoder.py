import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, 128, padding_idx=1)
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        # batch * seq (always 1) * vocab_size
        # (word word has more weights.)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):

        input = self.embedding(input)  # batch * embedding
        input = input.unsqueeze(1)  # batch * seq * embedding

        output, hidden = self.lstm(input, hidden)
        output = self.out(output)
        output = self.softmax(output)  # batch * seq * vocab_size (prob.)
        output = output.squeeze(dim=1)  # batch * vocab_size (prob.)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
