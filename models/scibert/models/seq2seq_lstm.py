import torch
import torch.nn as nn
import torch.nn.functional as F

from data_util.argparser import args
from models.encoder import EncoderLSTM
from models.decoder import DecoderLSTM
# from bert_serving.client import BertClient
from trainings.train_utils import get_input_from_batch, get_output_from_batch

use_cuda = args.use_gpu and torch.cuda.is_available()

class Seq2SeqLSTM(nn.Module):

    def __init__(self, hidden_size, vocab_size, max_length):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = EncoderLSTM(hidden_size, vocab_size)
        self.decoder = DecoderLSTM(hidden_size, vocab_size)
        #self.criterion = nn.NLLLoss(reduction='none')
        self.criterion = nn.NLLLoss()
        self.max_length = max_length

    def forward(self, batch):
        # enc_batch = batch.enc_batch
        enc_batch, enc_padding_mask, enc_lens = get_input_from_batch(batch, use_cuda=use_cuda)
        dec_batch, target_batch, dec_padding_mask, max_dec_len = get_output_from_batch(batch, use_cuda=use_cuda)
        batch_size = len(enc_batch)

        # Encoding sentences
        outputs, hidden_state = self.encoder(enc_batch)

        loss = 0
        teacher_forcing = torch.rand(1) > 0.5
        if teacher_forcing:
            # Does not take the output as the input
            for t in range(max(self.max_length, max_dec_len)):
                x = dec_batch[:, t]  # Batch * 1 (for each timestep)
                y = target_batch[:, t]
                output, hidden_state = self.decoder(x, hidden_state)  # Output: batch * vocab_size (prob.)
                F.dropout(hidden_state[0], p=0.2, inplace=True)
                F.dropout(hidden_state[1], p=0.2, inplace=True)

                # Compute the loss
                rows_loss = self.criterion(output, y)
                #batch_loss = (dec_padding_mask[:,t] * rows_loss).mean()  # Mask the output
                loss += rows_loss  # Update the total loss
        else:
            # use the output as the input
            x = dec_batch[:, 0]
            for t in range(max(self.max_length, max_dec_len)):
                y = target_batch[:, t]
                output, hidden_state = self.decoder(x, hidden_state)  # Output: batch * vocab_size (prob.)
                F.dropout(hidden_state[0], p=0.2, inplace=True)
                F.dropout(hidden_state[1], p=0.2, inplace=True)

                # Compute the loss
                rows_loss = self.criterion(output, y)
                #batch_loss = (dec_padding_mask[:,t] * rows_loss).mean()  # Mask the output
                loss += rows_loss  # Update the total loss

                x = torch.argmax(output, dim=1).detach()  # for the next input

        return loss
