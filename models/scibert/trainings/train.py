from __future__ import unicode_literals, print_function, division

import os
import time
import argparse

import tensorflow as tf
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad
import time
from shutil import copyfile
from data_util.batcher import Batcher
from data_util.data import Vocab, ids2words
from data_util.argparser import args
from trainings.train_utils import get_output_from_batch, calc_running_avg_loss, save_running_avg_loss

from models.decoder import DecoderLSTM
from bert_serving.client import BertClient

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
from models.bert_lstm import BertLSTMModel
from models.seq2seq_lstm import Seq2SeqLSTM

use_cuda = args.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(args.vocab_path, args.vocab_size)
        sys.stdout.flush()
        self.batcher = Batcher(args.train_data_path, self.vocab, mode='train',
                               batch_size=args.batch_size, single_pass=False)
        time.sleep(15)
        vocab_size = self.vocab.size()
        self.model = BertLSTMModel(args.hidden_size, self.vocab.size(), args.max_dec_steps)
        # self.model = Seq2SeqLSTM(args.hidden_size, self.vocab.size(), args.max_dec_steps)
        if use_cuda:
            self.model = self.model.cuda()

        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        train_logs = os.path.join(args.logs, "train_logs")
        eval_logs = os.path.join(args.logs, "eval_logs")
        self.train_summary_writer = tf.summary.FileWriter(train_logs)
        self.eval_summary_writer = tf.summary.FileWriter(eval_logs)


    def trainOneBatch(self, batch):
        self.model_optimizer.zero_grad()
        loss = self.model(batch)
        loss.backward()

        clip_grad_norm_(self.model.parameters(), 5)
        # clip_grad_norm_(self.model.decoder.parameters(), 5)
        self.model_optimizer.step()
        return loss.item() / args.max_dec_steps

    def trainIters(self):
        running_avg_loss = 0
        s_time = time.time()
        for t in range(args.max_iteration):
            batch = self.batcher.next_batch()
            loss = self.trainOneBatch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, decay=0.999)
            save_running_avg_loss(running_avg_loss, t, self.train_summary_writer)

            # Print every 100 steps
            if (t+1) % 1 == 0:
                time_run = time.time() - s_time
                s_time = time.time()
                print ("timestep: {}, loss: {}, time: {}s".format(t, running_avg_loss, time_run))
                sys.stdout.flush()

            # Save the model every 1000 steps
            if (t+1) % args.save_every_itr == 0:
                with torch.no_grad():
                    self.save_checkpoint(t, running_avg_loss)
                    self.model.eval()
                    self.evaluate(t)
                    self.model.train()


    def save_checkpoint(self, step, loss):
        checkpoint_file = "checkpoint_{}".format(step)
        checkpoint_path = os.path.join(args.logs, checkpoint_file)
        torch.save({
            'timestep': step,
            'model_state_dict': self.model.decoder.state_dict(),
            'optimizer_state_dict': self.model_optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        # torch.save({
        #     'timestep': step,
        #     'encoder_state_dict': self.model.encoder.state_dict(),
        #     'decoder_state_dict': self.model.decoder.state_dict(),
        #     'optimizer_state_dict': self.model_optimizer.state_dict(),
        #     'loss': loss
        # }, checkpoint_path)

    def evaluate(self, timestep):
        self.eval_batcher = Batcher(args.eval_data_path, self.vocab, mode='train',
                               batch_size=args.batch_size, single_pass=True)
        time.sleep(15)
        t1 = time.time()
        batch = self.eval_batcher.next_batch()
        running_avg_loss = 0
        while batch is not None:
            loss = self.model(batch)
            loss = loss / args.max_dec_steps
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss)
            batch = self.eval_batcher.next_batch()

        # Save the evaluation score
        time_spent = time.time() - t1
        print ("Evaluation Loss: {}, Time: {}s".format(running_avg_loss, time_spent))
        save_running_avg_loss(running_avg_loss, timestep, self.eval_summary_writer)
        sys.stdout.flush()

if __name__ == '__main__':
    train_processor = Train()
    train_processor.trainIters()
