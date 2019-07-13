import os

root_dir = os.path.expanduser("~")
project_folder = "/Users/vitou/Workspaces/AizawaLab/playground/scientific_question_generation/pytorch-pointer-generator/"
model_name = "pointer_network_001"
step_per_epoch = 50

data_folder = "/Users/vitou/Workspaces/AizawaLab/playground/scientific_question_generation/Text-Summarizer-Pytorch/data"
train_data_path = os.path.join(data_folder, "chunked/train/train_*.bin")
eval_data_path = os.path.join(data_folder, "chunked/valid/valid_*.bin")
decode_data_path = os.path.join(data_folder, "chunked/test/test_*.bin")
vocab_path = os.path.join(data_folder, "vocabs.txt")
logs = os.path.join(project_folder, "logs")
log_root = os.path.join(logs, model_name)
print (train_data_path)

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 4
max_enc_steps=150
max_dec_steps=15
beam_size=1
min_dec_steps=35
vocab_size=10000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15
