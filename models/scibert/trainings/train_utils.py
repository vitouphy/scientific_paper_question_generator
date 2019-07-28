from torch.autograd import Variable
import numpy as np
import torch
import tensorflow as tf


def get_input_from_batch(batch, use_cuda):
  batch_size = len(batch.enc_lens)
  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = batch.enc_lens

  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()

  return enc_batch, enc_padding_mask, enc_lens


def get_output_from_batch(batch, use_cuda=False):
  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  target_batch = Variable(torch.from_numpy(batch.target_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)

  if use_cuda:
    dec_batch = dec_batch.cuda()
    target_batch = target_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()

  return dec_batch, target_batch, dec_padding_mask, max_dec_len

def calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    # running_avg_loss = min(running_avg_loss, 12)  # clip
    return running_avg_loss

def save_running_avg_loss(running_avg_loss, step, summary_writer, decay=0.99):
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay={}'.format(decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
