import argparse
import os

parser = argparse.ArgumentParser(description="Parameters for BERT-LSTM")

# Data and Folder
parser.add_argument("--train_data_path", dest="train_data_path", required=False)
parser.add_argument("--eval_data_path", dest="eval_data_path", required=False,)
parser.add_argument("--decode_data_path", dest="decode_data_path", required=False)
parser.add_argument("--vocab_path", dest="vocab_path", required=True)
parser.add_argument("--logs", dest="logs", required=True)
parser.add_argument("--checkpoint", dest="checkpoint", required=False)
parser.add_argument("--output_name", dest="output_name", required=False)

# parameters
parser.add_argument("--max_iteration", dest="max_iteration", default=1000000)
parser.add_argument("--hidden_size", dest="hidden_size", default=768)
parser.add_argument("--emb_dim", dest="emb_dim", default=128)
parser.add_argument("--batch_size", dest="batch_size", default=16)
parser.add_argument("--max_enc_steps", dest="max_enc_steps", default=150)
parser.add_argument("--max_dec_steps", dest="max_dec_steps", default=15)
parser.add_argument("--beam_size", dest="beam_size", default=100)
parser.add_argument("--min_dec_steps", dest="min_dec_steps", default=35)
parser.add_argument("--vocab_size", dest="vocab_size", default=10000)
parser.add_argument("--lr", dest="lr", default=0.15)
parser.add_argument("--use_gpu", dest="use_gpu", default=True)

args = parser.parse_args()
