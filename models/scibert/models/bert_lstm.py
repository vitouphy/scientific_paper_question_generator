import torch
import torch.nn as nn
import torch.nn.functional as F

from data_util.argparser import args
from models.decoder import DecoderLSTM
from bert_serving.client import BertClient
from trainings.train_utils import get_output_from_batch

use_cuda = args.use_gpu and torch.cuda.is_available()

class BertLSTMModel(nn.Module):

    def __init__(self, hidden_size, vocab_size, max_length):
        super(BertLSTMModel, self).__init__()
        self.bertClient = BertClient()
        self.decoder = DecoderLSTM(hidden_size, vocab_size)
        self.criterion = nn.NLLLoss(reduction='none')
        self.max_length = max_length

    def forward(self, batch):
        enc_batch = batch.enc_batch
        dec_batch, dec_padding_mask, max_dec_len = get_output_from_batch(batch, use_cuda=use_cuda)
        batch_size = len(enc_batch)

        # Encoding sentences
        encoded_state = self.bertClient.encode(enc_batch, is_tokenized=True)
        h_0 = torch.Tensor(encoded_state).unsqueeze(0)  # batch_size * 768
        hidden_state = (h_0, h_0)

        loss = 0
        for t in range(max(self.max_length, max_dec_len)-1):
            # Pass info to decoder
            x = dec_batch[:, t]  # Batch * 1 (for each timestep)
            output, hidden_state = self.decoder(x, hidden_state)  # Output: batch * vocab_size (prob.)
            y = torch.LongTensor(dec_batch[:, t+1])  # y : batch * 1

            # Compute the loss
            rows_loss = self.criterion(output, y)
            batch_loss = (dec_padding_mask[:,t] * rows_loss).mean()  # Mask the output
            loss += batch_loss  # Update the total loss

        return loss
