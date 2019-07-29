from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import sys

import tensorflow as tf
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from shutil import copyfile
from data_util.batcher import Batcher
from data_util.data import Vocab, ids2words
from data_util.argparser import args
from trainings.train_utils import get_input_from_batch, get_output_from_batch, calc_running_avg_loss, save_running_avg_loss

from models.encoder import EncoderLSTM
from models.decoder import DecoderLSTM
from bert_serving.client import BertClient

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from models.bert_lstm import BertLSTMModel

use_cuda = args.use_gpu and torch.cuda.is_available()

class Decoder(object):

    def __init__(self):
        self.vocab = Vocab(args.vocab_path, args.vocab_size)
        self.batcher = Batcher(args.decode_data_path, self.vocab, mode='decode',
                               batch_size=1, single_pass=True)  # support only 1 item at a time
        time.sleep(15)
        vocab_size = self.vocab.size()
        self.beam_size = args.beam_size
        # self.bertClient = BertClient()
        self.encoder = EncoderLSTM(args.hidden_size, self.vocab.size())
        self.decoder = DecoderLSTM(args.hidden_size, self.vocab.size())
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        # Prepare the output folder and files
        output_dir = os.path.join(args.logs, "outputs")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, "decoder_{}.txt".format(args.output_name))
        self.file = open(output_file, "w+")

    def load_model(self, checkpoint_file):
        print ("Loading Checkpoint: ", checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        encoder_state_dict = checkpoint['encoder_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        self.encoder.eval()
        self.decoder.eval()
        print ("Weights Loaded")

    def decode(self):
        batch = self.batcher.next_batch()
        count = 0

        while batch is not None:
            # Preparing
            enc_batch, enc_padding_mask, enc_lens = get_input_from_batch(batch, use_cuda=use_cuda)
            dec_batch, target_batch, dec_padding_mask, max_dec_len = get_output_from_batch(batch, use_cuda=use_cuda)
            batch_size = len(enc_batch)
            proba_list = [0] * batch_size
            generated_list = [[]]

            # Encoding sentences
            outputs, hidden_state = self.encoder(enc_batch)

            x = torch.LongTensor([2]) # start sequence
            if use_cuda:
                x = x.cuda()

            """ Normal Approach """
            #answers = torch.ones((batch_size, args.max_dec_steps), dtype=torch.long)
            #for t in range(max(args.max_dec_steps, max_dec_len)):
            #    output, hidden_state = self.decoder(x, hidden_state)  # Output: batch * vocab_size (prob.)
            #    idx = torch.argmax(output, dim=1)
            #    answers[:, t] = idx.detach()
            #    x = idx

            """ Beam Approach """
            for t in range(max(args.max_dec_steps, max_dec_len)-1):
                output, hidden_state = self.decoder(x, hidden_state)

                # For each sentence, find b best answers (beam search)
                states = []  # (probab, generated, hidden_state_index)
                for i, each_decode in enumerate(output):
                    prev_proba = proba_list[i]
                    prev_generated = generated_list[i]
                    arr = each_decode.detach().cpu().numpy()  # log-probab of each word
                    indices = arr.argsort()[-self.beam_size:][::-1]  #index
                    for idx in indices:
                        proba = arr[idx] + proba_list[i]  # new probab and prev
                        generated = prev_generated.copy()
                        generated.append(idx)
                        states.append((proba, generated, i))

                # Sort for the best generated sequence among all
                states.sort(key=lambda x: x[0], reverse=True)

                # Variables
                new_proba_list = []
                new_generated = []
                new_hidden = torch.Tensor()
                new_cell = torch.Tensor()
                new_x = torch.LongTensor()

                if use_cuda:
                    new_hidden = new_hidden.cuda()
                    new_cell = new_cell.cuda()
                    new_x = new_x.cuda()

                # Select top b sequences
                for state in states[:self.beam_size]:
                    new_proba_list.append(state[0])
                    new_generated.append(state[1])
                    idx = state[2]

                    h_0 = hidden_state[0].squeeze(0)[idx].unsqueeze(0)
                    c_0 = hidden_state[1].squeeze(0)[idx].unsqueeze(0)
                    new_hidden = torch.cat((new_hidden, h_0), dim=0)
                    new_cell = torch.cat((new_cell, c_0), dim=0)
                    generated_idx = torch.LongTensor([state[1][-1]])
                    if use_cuda:
                        generated_idx = generated_idx.cuda()

                    new_x = torch.cat((new_x, generated_idx))

                # Save the list
                proba_list = new_proba_list
                generated_list = new_generated
                hidden_state = (new_hidden.unsqueeze(0), new_cell.unsqueeze(0))
                x = new_x
            
            # Convert from id to word
            # answer = answers[0].numpy()
            answer = new_generated[0]
            sentence = ids2words(answer, self.vocab)
            self.file.write("{}\n".format(sentence))
            print ("Writing line #{} to file ...".format(count+1))
            self.file.flush()
            sys.stdout.flush()

            count += 1
            batch = self.batcher.next_batch()


if __name__ == '__main__':
    decoder = Decoder()
    decoder.load_model(args.checkpoint)
    decoder.decode()
