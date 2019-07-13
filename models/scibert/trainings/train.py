from __future__ import unicode_literals, print_function, division

import os
import time
import argparse

import tensorflow as tf
import torch
# from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from shutil import copyfile
from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from models.decoder import DecoderLSTM
# from data_util.utils import calc_running_avg_loss
# from train_util import get_input_from_batch, get_output_from_batch
from bert_serving.client import BertClient



use_cuda = config.use_gpu and torch.cuda.is_available()

HIDDEN_SIZE = 768
OUTPUT_SIZE = 100

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        OUTPUT_SIZE = self.vocab.size()
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=4, single_pass=False)

        self.bertClient = BertClient()
        self.decoder = DecoderLSTM(HIDDEN_SIZE, OUTPUT_SIZE)

        # time.sleep(15)
        #
        # train_dir = os.path.join(config.log_root, 'train')
        # if not os.path.exists(train_dir):
        #     os.mkdir(train_dir)
        #
        # self.model_dir = os.path.join(train_dir, 'model')
        # if not os.path.exists(self.model_dir):
        #     os.mkdir(self.model_dir)

        # copy config file to the experiment folder
        # src_config = os.path.join(config.project_folder, "data_util/config.py")
        # dst_config = os.path.join(config.log_root, "config.py")
        # copyfile(src_config, dst_config)

        # self.summary_writer = tf.summary.FileWriter(train_dir)

    def trainOneBatch(self, batch):
        # Encoding sentences
        encoded_state = self.bertClient.encode(batch.enc_batch, is_tokenized=True, show_tokens=True)
        encoded_state = torch.Tensor(encoded_state)  # batch_size * 768
        print (encoded_state.size())

        # Decoding
        # For the first timestep
        # self.decoder(encoded_state, encoded_state)



    def trainIters(self):
        batch = self.batcher.next_batch()
        self.trainOneBatch(batch)
        # print (batch)
        # print (batch.enc_batch[0])
        # print (batch.dec_batch[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.trainIters()
    # train_processor.trainIters(config.max_iterations, args.model_file_path)
